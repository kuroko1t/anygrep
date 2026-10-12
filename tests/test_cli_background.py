"""Searches, `index`, `status`, `remove` and `reset` with a background index
(the job runs inside the test through conftest.sync_spawn, or is faked with a
state file and a writer lock held by the test). A search never starts a job:
only `anygrep index --background` does."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from anygrep import cli, job, thresholds
from anygrep.paths import canonical
from anygrep.store import Store, acquire_writer_lock

from .conftest import sync_spawn
from .test_cli_search_v01 import all_tty, db_path, index, run

# The job's line: its progress bar ("?" until it has measured its speed).
LINE = r"anygrep: background index: Indexing 4/10  40%\|.{10}\| \[00:0\d<\?\] \(anygrep status\)"


@pytest.fixture
def notes(corpus: Path) -> Path:
    (corpus / "notes" / "tax.md").write_text("# Tax return\n\nThe tax return is due.\n")
    return corpus


@pytest.fixture
def cutoff(monkeypatch) -> None:
    for pair in ("text->text", "text->pdf"):
        monkeypatch.setitem(thresholds.MIN_SCORES, pair, {0: 0.3})
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.3})


@pytest.fixture
def live_job(env, tmp_path: Path):
    """A running job (this process is its PID and holds the writer lock)."""
    locks = []

    def start(scope: list[Path], **fields) -> job.JobState:
        db_path().parent.mkdir(parents=True, exist_ok=True)
        st = job.JobState(
            pid=os.getpid(), started_at=time.time(), threads=2,
            scope=[str(canonical(p)) for p in scope], phase="images", files_total=10,
            files_done=4, text_total=2, cost_total=10.0, cost_done=4.0,
            log=str(job.job_paths(db_path()).log),
        )  # fmt: skip
        for k, v in fields.items():
            setattr(st, k, v)
        job.write_state(db_path(), st)
        locks.append(acquire_writer_lock(db_path()))
        return st

    yield start
    for lock in locks:
        lock.close()


def test_index_flag_combinations(env, notes: Path, capsys) -> None:
    code, _, err = run(capsys, "index", "--background", "--dry-run", str(notes))
    assert code == 2 and "not both" in err
    code, _, err = run(capsys, "index", "--stop", str(notes))
    assert code == 2 and "--stop takes no PATH" in err
    code, _, err = run(capsys, "bread", "--background", str(notes))
    assert code == 2 and "unrecognized arguments: --background" in err


@pytest.mark.parametrize("terminal", [False, True])
def test_a_search_never_starts_a_job(env, notes: Path, cutoff, capsys, monkeypatch, terminal):
    del sync_spawn.argvs[:]
    if terminal:
        all_tty(monkeypatch, "y\n")
    code, _, err = run(capsys, "bread", str(notes), "--index-budget", "0")
    assert sync_spawn.argvs == [] and not job.job_paths(db_path()).state.exists()
    assert code == (0 if terminal else 2) and "background" not in err


def test_a_job_for_another_folder(
    env, notes: Path, tmp_path: Path, cutoff, capsys, monkeypatch, live_job
) -> None:
    other = tmp_path / "pics"
    other.mkdir()
    live_job([other])
    all_tty(monkeypatch, "y\n")
    code, _, err = run(capsys, "bread", str(notes))
    where = cli._tilde(str(canonical(notes)))
    # Nothing indexed and the job holds the index: say how to add it, change nothing.
    assert code == 2 and err.strip().splitlines() == [
        f"anygrep: nothing under {where} is indexed yet; a background index of "
        f"{cli._tilde(str(other))} is running.",
        f"  add it to the job with: anygrep index --background {where}",
    ]
    assert job.read_state(db_path()).queue == [] and cli.sys.stdin.read() == "y\n"


def test_a_job_for_another_folder_partly_indexed(
    env, notes: Path, tmp_path: Path, cutoff, capsys, live_job
) -> None:
    index(capsys, notes)
    (notes / "new.md").write_text("bread rolls\n", encoding="utf-8")
    other = tmp_path / "pics"
    other.mkdir()
    live_job([other])
    code, out, err = run(capsys, "bread", str(notes), "-l")
    where = cli._tilde(str(canonical(notes)))
    assert code == 0 and "bread.md" in out and "new.md" not in out
    assert err.strip().splitlines() == [
        f"anygrep: a background index of {cli._tilde(str(other))} is running, so {where} "
        f"was not checked for new files; add it with: anygrep index --background {where}"
    ]


def test_live_job_progress_line(env, notes: Path, cutoff, capsys, live_job) -> None:
    index(capsys, notes)
    (notes / "new.md").write_text("bread rolls\n", encoding="utf-8")
    calls = env.calls["docs"]
    live_job([notes])
    code, out, err = run(capsys, "bread", str(notes), "-l")
    assert code == 0 and "new.md" not in out and env.calls["docs"] == calls  # no auto-index
    assert len(err.strip().splitlines()) == 1 and re.fullmatch(LINE, err.strip())
    code, out, err = run(capsys, "bread", "--everywhere", "-l")
    assert code == 0 and re.fullmatch(LINE, err.strip())
    code, out, err = run(capsys, "bread", str(notes), "-q")
    assert code == 0 and out == "" and err == ""


def test_live_job_planning_and_nothing_indexed(env, notes: Path, cutoff, capsys, live_job) -> None:
    live_job([notes], phase="planning")
    code, out, err = run(capsys, "bread", str(notes))
    where = cli._tilde(str(canonical(notes)))
    assert code == 2 and out == ""
    assert err.strip() == (
        f"anygrep: nothing under {where} is indexed yet; background index: "
        f"counting files under {where} (anygrep status)"
    )


def test_live_job_queues_an_uncovered_folder(
    env, notes: Path, tmp_path: Path, cutoff, capsys, live_job
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.md").write_text("bread\n", encoding="utf-8")
    index(capsys, other)
    live_job([notes])
    code, _, err = run(capsys, "index", "--background", str(other))
    assert code == 0
    assert job.read_state(db_path()).queue == [[str(canonical(other))]]
    assert err.strip() == "anygrep: " + (
        f"{cli._tilde(str(canonical(other)))} is queued for background indexing after "
        f"{cli._tilde(str(canonical(notes)))} (anygrep status)"
    )
    code, _, err = run(capsys, "bread", str(other))
    assert code in (0, 1) and err.strip().endswith("is queued for background indexing after "
        f"{cli._tilde(str(canonical(notes)))} (anygrep status)")  # fmt: skip


def test_status_block_and_json(env, notes: Path, tmp_path: Path, capsys, live_job) -> None:
    index(capsys, notes)
    st = live_job([notes], queue=[[str(tmp_path)]])
    code, out, _ = run(capsys, "status")
    assert code == 0
    where = cli._tilde(str(canonical(notes)))
    lines = out.splitlines()
    i = next(i for i, x in enumerate(lines) if x.startswith("Background:"))
    assert lines[i].startswith(f"Background: indexing {where}, PID {os.getpid()}, started ")
    assert lines[i].endswith(", 2 threads")
    assert re.fullmatch(
        r"  Indexing 4/10  40%\|.{10}\| \[00:0\d<\?\]  \(text and PDF done, images newest first\)",
        lines[i + 1],
    )
    assert lines[i + 2] == f"  then: {cli._tilde(str(tmp_path))}"
    assert lines[i + 3].startswith("  log: ") and lines[i + 3].endswith(
        "stop: anygrep index --stop"
    )
    info = json.loads(run(capsys, "status", "--json")[1])
    assert info["job"]["pid"] == st.pid and info["job"]["alive"] is True
    assert info["job"]["eta_seconds"] is None and info["job"]["queue"] == [[str(tmp_path)]]


def test_status_without_a_job(env, notes: Path, capsys) -> None:
    index(capsys, notes)
    code, out, _ = run(capsys, "status")
    assert code == 0 and "Background:" not in out
    assert json.loads(run(capsys, "status", "--json")[1])["job"] is None


def test_stale_job_is_reported_once(env, notes: Path, capsys) -> None:
    index(capsys, notes)
    st = job.JobState(pid=999_999_999, started_at=time.time() - 100, phase="images",
                      scope=[str(canonical(notes))], files_total=1363, files_done=340)  # fmt: skip
    job.write_state(db_path(), st)
    code, _, err = run(capsys, "bread", str(notes))
    where = cli._tilde(str(canonical(notes)))
    assert code in (0, 1)
    assert err.splitlines()[0] == (
        f"anygrep: the background index of {where} stopped before it finished "
        f"(340/1,363 files); resume with: anygrep index --background {where}"
    )
    assert not job.job_paths(db_path()).state.exists()
    assert "stopped before" not in run(capsys, "bread", str(notes))[2]


def test_stop_without_a_job(env, notes: Path, capsys) -> None:
    code, out, _ = run(capsys, "index", "--stop")
    assert code == 0 and out == "No background index is running.\n"
    index(capsys, notes)
    code, out, _ = run(capsys, "index", "--stop")
    assert code == 0 and out == "No background index is running.\n"


def test_writers_refused_while_a_job_runs(env, notes: Path, capsys, live_job) -> None:
    index(capsys, notes)
    live_job([notes])
    busy = (
        "anygrep: a background index is running (anygrep status); "
        "stop it with: anygrep index --stop"
    )
    code, _, err = run(capsys, "index", str(notes))
    assert code == 2 and err.splitlines() == [
        busy,
        "  or add the folder to it with: anygrep index --background PATH",
    ]
    assert run(capsys, "rm", str(notes)) == (2, "", busy + "\n")
    assert run(capsys, "reset", "-y") == (2, "", busy + "\n")
    # Options cannot change a running job; a folder it covers attaches to it.
    code, _, err = run(capsys, "index", "--background", "--no-images", str(notes))
    assert code == 2 and "options apply only to a new job" in err
    code, _, err = run(capsys, "index", "--background", str(notes / "notes"))
    assert code == 0 and "is already being indexed in the background" in err


def test_index_background_busy_with_a_foreground_writer(env, notes: Path, capsys) -> None:
    index(capsys, notes)
    lock = acquire_writer_lock(db_path())
    try:
        code, _, err = run(capsys, "index", "--background", str(notes))
        assert code == 2 and err.strip() == cli.BUSY
    finally:
        lock.close()


def test_index_background_all_roots_and_threads(env, notes: Path, capsys) -> None:
    index(capsys, notes)
    del sync_spawn.argvs[:]
    assert run(capsys, "index", "--background", "-j", "3")[0] == 0
    argv = sync_spawn.argvs[0][len(job.WORKER_PREFIX) :]
    assert argv[argv.index("-j") + 1] == "3" and argv[-1] == str(canonical(notes))
    st = Store(db_path(), create=False)
    assert st.counts()["text"]["ok"] == 3
    st.close()
