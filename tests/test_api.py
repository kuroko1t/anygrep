"""The Python interface (anygrep.Index) with the fake model."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import anygrep
from anygrep import cli, embedder, job, thresholds
from anygrep.store import acquire_writer_lock

from .conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cutoff(monkeypatch):
    """Cutoffs for the fake model: shared words pass, unrelated words do not."""
    for pair in ("text->text", "text->pdf"):
        monkeypatch.setitem(thresholds.MIN_SCORES, pair, {0: 0.3})
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.3})


@pytest.fixture
def loads(monkeypatch, env: FakeEmbedder) -> list[int]:
    """Index loads the fake model; records the threads of each model loader
    made (the real one loads the model once and keeps it)."""
    calls: list[int] = []

    def factory(threads: int, quiet: bool = True):
        assert quiet
        calls.append(threads)
        return lambda vision: env

    monkeypatch.setattr(embedder, "embedder_factory", factory)
    return calls


@pytest.fixture
def idx(tmp_path: Path, loads):
    with anygrep.Index(tmp_path / "api.sqlite", threads=2) as i:
        yield i


def test_import_is_cheap_and_turns_telemetry_off() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import os, anygrep; "
        "heavy = [m for m in ('numpy', 'onnxruntime', 'PIL', 'tokenizers') "
        "if m in sys.modules]; "
        "print(heavy, os.environ['ORT_DISABLE_TELEMETRY'], anygrep.__version__)"
    )
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.split() == ["[]", "1", anygrep.__version__], out.stderr


def test_public_names() -> None:
    for name in anygrep.__all__:
        assert getattr(anygrep, name) is not None
    for err in (
        "IndexAccessError",
        "IndexBusyError",
        "IndexMismatchError",
        "ModelUnavailableError",
        "ModelLoadError",
    ):
        assert issubclass(getattr(anygrep, err), anygrep.AnygrepError)
    # Code that caught the old RuntimeError subclasses keeps working.
    assert issubclass(anygrep.IndexMismatchError, RuntimeError)


def test_update_and_search(idx: anygrep.Index, corpus: Path, cutoff, loads, capfd) -> None:
    assert idx.search("sourdough bread") == []  # nothing indexed yet
    r = idx.update(corpus)
    assert (r.new, r.updated, r.unchanged, r.removed, r.failed) == (5, 0, 0, 0, 0)
    assert r.seconds >= 0
    assert idx.roots == [str(corpus.resolve())]

    hits = idx.search("sourdough bread flour", k=3)
    top = hits[0]
    assert Path(top.path).name == "bread.md" and top.type == "text"
    assert top.line_start == 1 and top.line_end and top.page is None
    assert "Sourdough" in top.snippet and "Sourdough" in (top.text or "")
    assert not top.weak and 0 < top.score <= 1
    assert all(not h.weak for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    pdf = idx.search("reset the wifi password", type="pdf")
    assert [Path(h.path).name for h in pdf] == ["manual.pdf"] and pdf[0].page == 2
    assert idx.search("reset the wifi password", type=["text", "image"]) == []
    assert idx.search("sourdough bread", paths=corpus / "notes", k=1)[0].path == top.path
    assert idx.search("sourdough bread", paths=[str(corpus / "notes" / "bikes.txt")]) == []
    assert idx.search("sourdough bread", glob="*.txt") == []
    assert idx.search("sourdough bread", glob=["!notes/*"], type="text") == []

    # The model is loaded once, with the threads given, and reused.
    assert loads == [2]
    # Nothing is printed.
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_weak_results(idx: anygrep.Index, corpus: Path, cutoff) -> None:
    idx.update(corpus)
    assert idx.search("qwzx plorb fnord") == []
    weak = idx.search("qwzx plorb fnord", weak=True, k=3)
    assert len(weak) == 3 and all(h.weak for h in weak)


def test_same_results_as_cli_json(env, idx: anygrep.Index, corpus: Path, capsys) -> None:
    (corpus / "archive").mkdir()
    shutil.copy(corpus / "notes" / "bread.md", corpus / "archive" / "bread.md")
    idx.update(corpus)
    db = str(idx.path)
    for extra, kwargs in (
        ([], {}),
        (["--no-collapse"], {"collapse": False}),
        (["--chunks"], {"chunks": True}),
        (["-t", "pdf", "-t", "text", "-m", "3"], {"type": ["pdf", "text"], "k": 3}),
    ):
        argv = ["--db", db, "--json", "--weak", "--everywhere", *extra, "bread oven wifi"]
        assert cli.main(argv) == 0
        want = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        got = idx.search("bread oven wifi", weak=True, **kwargs)
        assert [(h.path, round(h.score, 4), h.weak, h.page, h.line_start) for h in got] == [
            (w["path"], w["score"], w["weak"], w["page"], w["line_start"]) for w in want
        ]
        assert [[x.path for x in h.similar] for h in got] == [
            [x["path"] for x in w["similar"]] for w in want
        ]
        assert [h.snippet for h in got] == [w["snippet"] or "" for w in want]
    top = idx.search("sourdough bread", weak=True)[0]
    assert len(top.similar) == 1 and Path(top.similar[0].path).name == "bread.md"


def test_incremental_update(idx: anygrep.Index, corpus: Path) -> None:
    idx.update(corpus)
    (corpus / "notes" / "bikes.txt").unlink()
    (corpus / "notes" / "new.md").write_text("A new note about tea.\n", encoding="utf-8")
    r = idx.update()  # every indexed folder
    assert (r.new, r.unchanged, r.removed) == (1, 4, 1)
    assert idx.update(str(corpus)).unchanged == 5


def test_progress_and_interruption(idx: anygrep.Index, corpus: Path) -> None:
    seen: list[tuple[int, int]] = []
    idx.update(corpus / "notes", progress=lambda done, total: seen.append((done, total)))
    assert seen[0] == (0, 2) and seen[-1] == (2, 2)

    class Stop(Exception):
        pass

    def stop_after_one(done: int, total: int) -> None:
        if done == 1:
            raise Stop

    with pytest.raises(Stop):
        idx.update(corpus, progress=stop_after_one)
    # The finished file was kept; the next run indexes the rest.
    r = idx.update(corpus)
    assert r.new == 2 and r.unchanged == 3


def test_ctrl_c_keeps_finished_files(idx: anygrep.Index, corpus: Path, env) -> None:
    env.fail_after = 1
    with pytest.raises(KeyboardInterrupt):
        idx.update(corpus)
    env.fail_after = None
    r = idx.update(corpus)
    assert r.new + r.unchanged == 5 and r.unchanged >= 1


def test_busy_index(idx: anygrep.Index, corpus: Path, monkeypatch) -> None:
    idx.update(corpus / "notes")
    lock = acquire_writer_lock(idx.path)
    try:
        with pytest.raises(anygrep.IndexBusyError, match="another anygrep process"):
            idx.update(corpus)
        assert idx.search("sourdough bread", weak=True)  # searching still works
    finally:
        lock.close()

    running = job.JobState(pid=os.getpid(), scope=[str(corpus)])
    monkeypatch.setattr(job, "check", lambda index, clean=True: (running, None))
    with pytest.raises(anygrep.IndexBusyError, match="background index"):
        idx.update(corpus)
    monkeypatch.undo()
    assert idx.update(corpus).new == 3  # the lock was given back


def test_image_query(idx: anygrep.Index, corpus: Path, tmp_path: Path) -> None:
    Image.new("RGB", (120, 90), (230, 10, 10)).save(corpus / "red-copy.png")
    idx.update(corpus)
    hits = idx.search(image=corpus / "red.png", weak=True, type="image", collapse=False)
    names = [Path(h.path).name for h in hits]
    assert names[0] == "red-copy.png" and "red.png" not in names  # the query itself is left out
    pil = idx.search(image=Image.new("RGBA", (64, 64), (0, 0, 250, 255)), weak=True, k=1)
    assert Path(pil[0].path).name == "blue.jpg" and pil[0].snippet == ""


def test_default_index_is_the_cli_one(env, loads, corpus: Path, cutoff, capsys) -> None:
    with anygrep.Index() as idx:
        assert idx.path == cli.default_index_path().resolve()
        idx.update(corpus)
    assert cli.main(["sourdough bread", "--everywhere", "-l"]) == 0
    assert "bread.md" in capsys.readouterr().out


def test_arguments_and_errors(idx: anygrep.Index, corpus: Path, tmp_path: Path) -> None:
    for bad in (
        lambda: idx.search(),
        lambda: idx.search("a", image=corpus / "red.png"),
        lambda: idx.search("  "),
        lambda: idx.search("a", k=0),
        lambda: idx.search("a", type="audio"),
        lambda: idx.update(corpus, image_tokens=100),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(FileNotFoundError):
        idx.update(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        idx.search("a", paths=tmp_path / "missing")
    assert idx.update() == anygrep.UpdateResult()  # nothing indexed: nothing to do

    other = tmp_path / "other.sqlite"
    other.write_text("not a database index", encoding="utf-8")
    with pytest.raises(anygrep.IndexMismatchError, match="not an anygrep index"):
        anygrep.Index(other)
    with pytest.raises(ValueError):
        anygrep.Index(tmp_path / "x.sqlite", threads=0)

    idx.close()
    idx.close()
    assert "closed" in repr(idx)
    with pytest.raises(ValueError, match="closed"):
        idx.search("a")


def test_empty_lists_do_not_mean_everything(idx: anygrep.Index, corpus: Path, cutoff) -> None:
    idx.update(corpus)
    for bad in (
        lambda: idx.search("sourdough bread", paths=[]),
        lambda: idx.search("sourdough bread", paths=[""]),
        lambda: idx.search("sourdough bread", glob=[]),
        lambda: idx.search("sourdough bread", type=[]),
        lambda: idx.update([]),
    ):
        with pytest.raises(ValueError, match=r"empty|type must be"):
            bad()
    assert idx.search("sourdough bread", paths=None, glob=None)


def test_paths_and_threads_setting(tmp_path: Path, loads, monkeypatch) -> None:
    with pytest.raises(ValueError, match="empty"):
        anygrep.Index("")
    with pytest.raises(anygrep.IndexAccessError, match="folder"):
        anygrep.Index(tmp_path)
    missing = tmp_path / "typo" / "index.sqlite"
    with pytest.raises(FileNotFoundError):
        anygrep.Index(missing, create=False)
    assert not missing.parent.exists()
    with anygrep.Index(missing) as idx:  # created, with its folder
        assert idx.roots == []
    with anygrep.Index(missing, create=False) as idx:
        assert idx.search("anything") == []
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(anygrep.IndexAccessError):
        anygrep.Index(blocker / "index.sqlite")

    monkeypatch.setenv("ANYGREP_THREADS", "abc")
    with pytest.raises(ValueError, match=r"ANYGREP_THREADS must be .* got 'abc'"):
        anygrep.Index(missing)
    with anygrep.Index(missing, threads=2) as idx:
        assert idx.threads == 2


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs file permissions")
def test_read_only_index(tmp_path: Path, corpus: Path, loads, cutoff) -> None:
    db = tmp_path / "ro" / "index.sqlite"
    with anygrep.Index(db) as idx:
        idx.update(corpus / "notes")
    db.chmod(0o444)
    try:
        with anygrep.Index(db) as idx:
            assert idx.search("sourdough bread")
            with pytest.raises(anygrep.IndexAccessError, match="read-only"):
                idx.update(corpus)
    finally:
        db.chmod(0o600)
    db.parent.chmod(0o555)
    try:
        with anygrep.Index(db) as idx:
            assert idx.search("sourdough bread")
            with pytest.raises(anygrep.AnygrepError):
                idx.update(corpus)
    finally:
        db.parent.chmod(0o700)


@pytest.mark.skipif(os.name != "posix", reason="Windows does not delete or replace open files")
def test_index_deleted_or_replaced_while_open(
    tmp_path: Path, corpus: Path, loads, cutoff, capsys
) -> None:
    db = tmp_path / "index.sqlite"
    with anygrep.Index(db) as idx:
        idx.update(corpus / "notes")
        assert idx.search("sourdough bread")
        assert cli.main(["--db", str(db), "reset", "-y"]) == 0
        # The next call opens the file that is there now (a new, empty one)
        # instead of writing to the deleted file.
        assert idx.search("sourdough bread") == []
        assert idx.roots == []
        r = idx.update(corpus / "notes")
        assert r.new == 2
    with anygrep.Index(db) as other:
        assert other.search("sourdough bread")

    # Replaced by another index.
    copy = tmp_path / "copy.sqlite"
    with anygrep.Index(copy) as c:
        c.update(corpus / "manual.pdf")
    with anygrep.Index(db) as idx:
        assert idx.search("sourdough bread")
        for s in ("-wal", "-shm"):
            (tmp_path / f"index.sqlite{s}").unlink(missing_ok=True)
        os.replace(copy, db)
        assert idx.search("sourdough bread") == []
        assert idx.roots == [str((corpus / "manual.pdf").resolve())]

    # Deleted, and create=False: the error says so.
    with anygrep.Index(db, create=False) as idx:
        db.unlink()
        with pytest.raises(FileNotFoundError):
            idx.search("sourdough bread")


def test_shared_between_threads(idx: anygrep.Index, corpus: Path, cutoff, loads) -> None:
    import threading

    idx.update(corpus)
    results: list[object] = []

    def work() -> None:
        try:
            results.append(idx.search("sourdough bread", k=1)[0].path)
            results.append(idx.update().unchanged)
        except BaseException as e:
            results.append(e)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8
    assert all(isinstance(r, (str, int)) for r in results), results
    assert loads == [2]  # one model for all threads


def test_image_tokens_and_extensions(idx: anygrep.Index, corpus: Path, cutoff) -> None:
    (corpus / "script.py").write_text("print('sourdough bread starter')\n", encoding="utf-8")
    idx.update(corpus, extensions=[".py"])
    assert any(h.path.endswith("script.py") for h in idx.search("sourdough bread starter"))
    r = idx.update(corpus, image_tokens=70)
    assert r.updated == 2  # the two images, embedded again
    r = idx.update(corpus, images=False)
    assert (r.unchanged, r.removed) == (4, 0)  # .py is remembered; images stay indexed
    assert len(idx.search("red picture", type="image", weak=True)) == 2
