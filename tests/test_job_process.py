"""Background indexing with a real detached worker process (the fake model,
tests/bg_worker.py): start, search while it runs, stop, resume, and a worker
killed with SIGKILL."""

from __future__ import annotations

import os
import re
import signal
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from anygrep import cli, job, thresholds
from anygrep.store import Store

from .conftest import FakeEmbedder, write_pdf

WORKER = Path(__file__).resolve().parent / "bg_worker.py"
IMAGES = 40


def make_images(root: Path, n: int, start: int = 0) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        im = Image.new("RGB", (64, 64), (200, (i * 37) % 256, (i * 11) % 256))
        im.putpixel((0, 0), (i % 256, i // 256, 7))  # every file is different
        im.save(root / f"img{i:03d}.png")
        os.utime(root / f"img{i:03d}.png", (1_700_000_000 + i, 1_700_000_000 + i))


@pytest.fixture
def bg(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANYGREP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANYGREP_DB", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # Slow enough that the job is still running when step 3 stops it, also on
    # a slow Windows runner (40 images: 40 s).
    monkeypatch.setenv("ANYGREP_TEST_IMAGE_DELAY", "1.0")
    monkeypatch.setattr(job, "WORKER_PREFIX", [sys.executable, str(WORKER)])
    monkeypatch.setattr(cli, "_prefetch_model", lambda vision, quiet: None)
    monkeypatch.setattr(cli, "_make_embedder_factory", lambda args: lambda vision: FakeEmbedder())
    for pair in ("text->text", "text->pdf"):
        monkeypatch.setitem(thresholds.MIN_SCORES, pair, {0: 0.3})
    corpus = tmp_path / "corpus"
    (corpus / "notes").mkdir(parents=True)
    for i, topic in enumerate(["bread", "bikes", "tax", "garden", "music"]):
        (corpus / "notes" / f"{topic}.md").write_text(
            f"# {topic}\n\nNotes about {topic} number {i}.\n", encoding="utf-8"
        )
    write_pdf(corpus / "manual.pdf", [["Router manual", "Plug in the router."]])
    make_images(corpus / "pics", IMAGES)
    db = tmp_path / "db" / "index.sqlite"
    spawned: list[int] = []
    real_spawn = job.spawn

    def spawn(*a):
        pid = real_spawn(*a)
        spawned.append(pid)
        return pid

    monkeypatch.setattr(job, "spawn", spawn)
    yield corpus, db, spawned
    for pid in spawned:  # only the workers this test started
        if job.pid_running(pid):
            os.kill(pid, signal.SIGKILL)
            end = time.monotonic() + 5
            while job.pid_running(pid) and time.monotonic() < end:
                time.sleep(0.05)


def wait_for(cond, timeout: float = 20.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


def counts(db: Path) -> dict:
    st = Store(db, create=False, readonly=True)
    try:
        return st.counts()
    finally:
        st.close()


def ok_files(db: Path) -> int:
    return sum(c.get("ok", 0) for c in counts(db).values())


def test_background_job_process(bg, capsys, monkeypatch) -> None:
    corpus, db, spawned = bg
    jp = job.job_paths(db)

    # 1. Start: detached, in its own session, at a lower priority.
    assert cli.main(["index", "--background", "--db", str(db), str(corpus)]) == 0
    assert "in the background" in capsys.readouterr().err
    wait_for(lambda: (s := job.read_state(db)) is not None and s.files_done >= 8)
    wait_for(lambda: ok_files(db) >= 6)  # the text is committed
    st = job.read_state(db)
    assert st.pid == spawned[0] and st.pid != os.getpid() and job.pid_running(st.pid)
    assert st.phase == "images" and st.text_total == 6
    if os.name == "posix":
        assert os.getsid(st.pid) != os.getsid(0)
        assert os.getpriority(os.PRIO_PROCESS, st.pid) >= min(19, os.nice(0) + job.NICE)

    # 2. A search meanwhile finds the text and ends with the progress line.
    assert cli.main(["bread", str(corpus), "--db", str(db), "-l"]) == 0
    cap = capsys.readouterr()
    assert "bread.md" in cap.out
    assert re.match(
        r"anygrep: background index: Indexing \d+/46 +\d+%\|.{10}\| \[[\d:]+<[\d:?]+\] "
        r"\(anygrep status --db .+\)$",
        cap.err.strip().splitlines()[-1],
    )
    assert cli.main(["status", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert f"PID {st.pid}" in out and "text and PDF done, images newest first" in out

    # 3. Stop: the worker exits, committed files stay, nothing is left behind.
    t0 = time.monotonic()
    assert cli.main(["index", "--stop", "--db", str(db)]) == 0
    assert time.monotonic() - t0 < 10
    cap = capsys.readouterr()
    assert cap.out.startswith("Stopped the background index of ")
    assert "stopping the background index..." in cap.err
    wait_for(lambda: not job.pid_running(st.pid), 5)
    assert not jp.state.exists() and not jp.stop.exists() and not job.lock_held(db)
    assert not jp.final.exists()
    assert 6 < ok_files(db) < 46
    # The count said is the files indexed, not the last saved progress.
    assert f" at {ok_files(db)}/46 files" in cap.out
    assert "Stopped after" in jp.log.read_text(encoding="utf-8")

    # 4. Resume: only the remaining images are embedded.
    monkeypatch.setenv("ANYGREP_TEST_IMAGE_DELAY", "0.1")
    assert cli.main(["index", "--background", "--db", str(db), str(corpus)]) == 0
    wait_for(lambda: not jp.state.exists(), 30)
    wait_for(lambda: not job.pid_running(spawned[-1]), 5)
    assert counts(db)["image"]["ok"] == IMAGES and ok_files(db) == 46
    log = jp.log.read_text(encoding="utf-8")
    assert sum(map(int, re.findall(r"image embeds: (\d+)", log))) == IMAGES
    capsys.readouterr()

    # 5. A killed worker is reported once, cleaned up, and resumes.
    if os.name != "posix":
        return
    make_images(corpus / "pics", 20, start=IMAGES)
    monkeypatch.setenv("ANYGREP_TEST_IMAGE_DELAY", "0.5")
    assert cli.main(["index", "--background", "--db", str(db)]) == 0  # all roots
    wait_for(lambda: (s := job.read_state(db)) is not None and s.files_done >= 1)
    time.sleep(1.0)
    monkeypatch.setenv("ANYGREP_TEST_IMAGE_DELAY", "0.05")
    pid = job.read_state(db).pid
    os.kill(pid, signal.SIGKILL)
    wait_for(lambda: not job.pid_running(pid), 5)
    capsys.readouterr()
    assert cli.main(["status", "--db", str(db)]) == 0
    err = capsys.readouterr().err
    assert "stopped before it finished" in err and "resume with: anygrep index --background" in err
    assert not jp.state.exists()
    assert cli.main(["status", "--db", str(db)]) == 0
    assert "stopped before" not in capsys.readouterr().err  # reported once
    assert cli.main(["index", "--background", "--db", str(db), str(corpus)]) == 0
    wait_for(lambda: not jp.state.exists(), 30)
    assert counts(db)["image"]["ok"] == IMAGES + 20
