from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from anygrep.errors import ModelLoadError
from anygrep.indexer import IndexOptions, run_index
from anygrep.search import scope_matcher, search
from anygrep.store import Store

from .conftest import FakeEmbedder


def index(store: Store, roots, emb: FakeEmbedder, **kw):
    opts = IndexOptions(quiet=True, **kw)
    return run_index(store, list(roots), opts, lambda vision: emb)


def test_full_then_incremental(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [corpus], fake_embedder)
    assert (rep.new, rep.errors, rep.skipped) == (5, 0, 0)
    counts = store.counts()
    assert counts["text"]["ok"] == 2
    assert counts["pdf"]["ok"] == 1 and counts["pdf"]["chunks"] == 2
    assert counts["image"]["ok"] == 2

    # nothing changed
    calls = dict(fake_embedder.calls)
    rep = index(store, [corpus], fake_embedder)
    assert rep.unchanged == 5 and rep.new == 0
    assert fake_embedder.calls == calls

    # mtime-only change: hash decides, nothing re-embedded
    p = corpus / "notes" / "bikes.txt"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    rep = index(store, [corpus], fake_embedder)
    assert rep.unchanged == 5 and fake_embedder.calls == calls

    # content change
    p.write_text("Completely different text about gardening and tomatoes.\n")
    rep = index(store, [corpus], fake_embedder)
    assert rep.updated == 1

    # deletion is pruned
    (corpus / "blue.jpg").unlink()
    rep = index(store, [corpus], fake_embedder)
    assert rep.removed == 1
    assert store.counts()["image"]["ok"] == 1


def test_search_ranks_and_filters(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    q = fake_embedder.embed_query("reset the wifi password")
    hits = search(store, q, n=3)
    assert hits[0].path.endswith("manual.pdf") and hits[0].page == 2
    hits = search(store, fake_embedder.embed_query("red picture"), n=1, kinds=["image"])
    assert hits[0].path.endswith("red.png")
    hits = search(store, q, n=10, scope=scope_matcher([], ["*.md"]))
    assert [Path(h.path).name for h in hits] == ["bread.md"]
    hits = search(store, q, n=10, scope=scope_matcher([str(corpus / "notes")], []))
    assert {Path(h.path).name for h in hits} == {"bread.md", "bikes.txt"}
    # one result per file unless chunks are requested
    all_chunks = search(store, q, n=10, kinds=["pdf"], per_file=False)
    assert len(all_chunks) == 2
    assert len(search(store, q, n=10, kinds=["pdf"])) == 1


def test_interrupt_and_resume(tmp_path: Path, corpus: Path) -> None:
    for i in range(6):
        Image.new("RGB", (64, 64), (10 * i, 200, 30)).save(corpus / f"g{i}.png")
    store = Store(tmp_path / "idx.sqlite")
    emb = FakeEmbedder()
    emb.fail_after = 4  # text batch + a few images, then Ctrl-C
    rep = index(store, [corpus], emb)
    assert rep.interrupted
    done = sum(v.get("ok", 0) for v in store.counts().values())
    assert 0 < done < 11

    emb2 = FakeEmbedder()
    rep = index(store, [corpus], emb2)
    assert not rep.interrupted
    assert rep.new == 11 - done
    assert sum(v.get("ok", 0) for v in store.counts().values()) == 11


def test_budget_change_reembeds_images(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder, image_budget=140)
    before = fake_embedder.calls["image"]
    rep = index(store, [corpus], fake_embedder, image_budget=70)
    assert rep.updated == 2
    assert fake_embedder.calls["image"] == before + 2


def test_bad_files_recorded_not_retried(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    (corpus / "broken.png").write_bytes(b"\x89PNG not really")
    (corpus / "tiny.png").write_bytes(b"")  # empty files are not even considered
    Image.new("RGB", (8, 8)).save(corpus / "icon.png")
    (corpus / "scan.pdf").write_bytes(b"%PDF-1.4 garbage")
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [corpus], fake_embedder)
    assert rep.errors == 2  # broken.png, scan.pdf
    assert rep.skipped == 1  # icon.png too small
    problems = {Path(p).name: (s, n) for p, s, n in store.problem_files()}
    assert problems["icon.png"][0] == "skipped"
    assert problems["broken.png"][0] == "error"
    rep = index(store, [corpus], fake_embedder)
    assert rep.errors == 0 and rep.unchanged == 8


def test_no_images_keeps_existing(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [corpus], fake_embedder, images=False)
    assert rep.deferred_images == 2
    assert "image" not in store.counts()
    index(store, [corpus], fake_embedder)
    rep = index(store, [corpus], fake_embedder, images=False)
    assert rep.removed == 0 and store.counts()["image"]["ok"] == 2


def test_store_survives_reopen(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    path = tmp_path / "idx.sqlite"
    store = Store(path)
    index(store, [corpus], fake_embedder)
    store.close()
    time.sleep(0.01)
    store = Store(path, create=False)
    assert store.roots() == [str(corpus)]
    assert store.counts()["text"]["ok"] == 2


needs_posix_perms = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)


class PoisonEmbedder(FakeEmbedder):
    """Fails on any batch that contains the word POISON."""

    def embed_documents(self, texts, titles):
        if any("POISON" in t for t in texts):
            raise RuntimeError("model failed on this input")
        return super().embed_documents(texts, titles)


def test_failing_batch_only_marks_the_bad_file(tmp_path: Path, corpus: Path) -> None:
    for i in range(40):
        (corpus / f"n{i:02d}.md").write_text(f"note number {i} about gardening\n")
    (corpus / "bad.md").write_text("POISON text\n")
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [corpus], PoisonEmbedder())
    assert rep.errors == 1
    assert store.counts()["text"]["ok"] == 42
    problems = {Path(p).name: s for p, s, _ in store.problem_files()}
    assert problems == {"bad.md": "error"}


def test_unexpected_error_keeps_finished_files(tmp_path: Path, corpus: Path) -> None:
    class Broken(FakeEmbedder):
        def embed_image(self, image):
            raise ModelLoadError("vision model cannot be loaded")

    store = Store(tmp_path / "idx.sqlite")
    with pytest.raises(ModelLoadError):
        index(store, [corpus], Broken())
    counts = store.counts()
    assert counts["text"]["ok"] == 2 and counts["pdf"]["ok"] == 1
    assert "image" not in counts  # not recorded as failed either


@needs_posix_perms
def test_unreadable_folder_keeps_its_indexed_files(tmp_path: Path, corpus: Path, fake_embedder):
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    notes = corpus / "notes"
    notes.chmod(0o300)  # cannot be listed
    try:
        rep = index(store, [corpus], fake_embedder)
    finally:
        notes.chmod(0o755)
    assert rep.removed == 0 and rep.walk.unreadable_dirs == [notes]
    assert store.counts()["text"]["ok"] == 2
    (corpus / "sub").mkdir()
    (corpus / "sub" / "x.md").write_text("x")
    (corpus / "sub").chmod(0)
    try:
        rep = index(store, [corpus], fake_embedder)  # must not raise
    finally:
        (corpus / "sub").chmod(0o755)
    assert rep.errors == 0


@pytest.mark.parametrize("root_is_file", [False, True])
@pytest.mark.parametrize("failure", [PermissionError, OSError])
def test_root_stat_failure_after_validation_keeps_index(
    tmp_path: Path, corpus: Path, fake_embedder, monkeypatch, root_is_file, failure
) -> None:
    from anygrep import indexer

    root = corpus / "notes" / "bread.md" if root_is_file else corpus
    store = Store(tmp_path / "idx.sqlite")
    index(store, [root], fake_embedder)
    before = store.counts()
    real_walk = indexer.walk
    real_stat = Path.stat

    def fail_root_stat(path, *args, **kwargs):
        if path == root:
            raise failure("temporary stat failure")
        return real_stat(path, *args, **kwargs)

    def walk_after_validation(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", fail_root_stat)
            yield from real_walk(*args, **kwargs)

    monkeypatch.setattr(indexer, "walk", walk_after_validation)
    rep = index(store, [root], fake_embedder)
    assert rep.removed == 0
    assert rep.walk.unreadable_dirs == [root]
    assert store.counts() == before


def test_deleted_file_root_is_pruned(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    root = corpus / "notes" / "bread.md"
    store = Store(tmp_path / "idx.sqlite")
    index(store, [root], fake_embedder)
    root.unlink()
    rep = index(store, [root], fake_embedder)
    assert rep.removed == 1
    assert rep.walk.missing_roots == [root]
    assert rep.walk.unreadable_dirs == []
    assert store.get_file(str(root)) is None


def test_directory_root_missing_after_validation_keeps_index(
    tmp_path: Path, corpus: Path, fake_embedder, monkeypatch
) -> None:
    from anygrep import indexer

    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    before = store.counts()
    real_walk = indexer.walk
    real_stat = Path.stat

    def missing_during_walk(path, *args, **kwargs):
        if path == corpus:
            raise FileNotFoundError(str(corpus))
        return real_stat(path, *args, **kwargs)

    def walk_after_validation(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(Path, "stat", missing_during_walk)
            yield from real_walk(*args, **kwargs)

    monkeypatch.setattr(indexer, "walk", walk_after_validation)
    rep = index(store, [corpus], fake_embedder)
    assert rep.removed == 0
    assert rep.walk.missing_roots == [corpus]
    assert rep.walk.unreadable_dirs == [corpus]
    assert store.counts() == before


@needs_posix_perms
def test_permission_error_is_retried(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    p = corpus / "locked.md"
    p.write_text("locked note\n")
    p.chmod(0)
    store = Store(tmp_path / "idx.sqlite")
    try:
        rep = index(store, [corpus], fake_embedder)
        assert rep.errors == 1
    finally:
        p.chmod(0o644)  # changes only the ctime
    rep = index(store, [corpus], fake_embedder)
    assert rep.updated == 1 and rep.errors == 0
    assert not store.problem_files()


def test_symlinked_root_is_not_indexed_twice(tmp_path: Path, corpus: Path, fake_embedder):
    link = tmp_path / "link"
    try:
        link.symlink_to(corpus, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not available")
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    rep = index(store, [link], fake_embedder)
    assert rep.new == 0 and rep.unchanged == 5
    assert store.roots() == [str(corpus)]
    assert sum(v.get("ok", 0) for v in store.counts().values()) == 5


def test_max_size_keeps_larger_indexed_files(tmp_path: Path, corpus: Path, fake_embedder):
    (corpus / "big.md").write_text("big " * 1000)
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    rep = index(store, [corpus], fake_embedder, max_size={"text": 1000})
    assert rep.removed == 0
    assert store.get_file(str(corpus / "big.md")) is not None


def test_single_file_root_is_stored(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.sqlite")
    f = corpus / "notes" / "bread.md"
    index(store, [f, corpus / "unknown.xyz"], fake_embedder)
    assert store.roots() == [str(f)]


class FlakyEmbedder(FakeEmbedder):
    """Fails like ONNX Runtime under memory pressure for the first `fails` calls."""

    def __init__(self, fails: int) -> None:
        super().__init__()
        self.fails = fails

    def _flaky(self) -> None:
        if self.fails > 0:
            self.fails -= 1
            raise RuntimeError("[ONNXRuntimeError] : 6 : Failed to allocate memory")

    def embed_documents(self, texts, titles):
        self._flaky()
        return super().embed_documents(texts, titles)

    def embed_image(self, image):
        self._flaky()
        return super().embed_image(image)


def test_embedding_failures_are_retried_next_run(tmp_path: Path, corpus: Path) -> None:
    store = Store(tmp_path / "idx.sqlite")
    # The text/pdf batch fails, then each of its 3 files alone, then 1 image.
    rep = index(store, [corpus], FlakyEmbedder(fails=5))
    assert rep.errors == 4 and rep.new == 1
    notes = [n for _, _, n in store.problem_files()]
    assert all(n.startswith("EmbedError: RuntimeError") for n in notes)
    rep = index(store, [corpus], FakeEmbedder())
    assert (rep.updated, rep.errors, rep.unchanged) == (4, 0, 1)
    assert not store.problem_files()


def test_moved_files_reuse_their_rows(tmp_path: Path, corpus: Path) -> None:
    emb = FakeEmbedder()
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], emb)
    calls = dict(emb.calls)
    (corpus / "pics").mkdir()
    (corpus / "red.png").rename(corpus / "pics" / "renamed-red.png")  # image: any new name
    (corpus / "notes" / "bikes.txt").rename(corpus / "pics" / "bikes.txt")  # same name
    (corpus / "notes" / "bread.md").rename(corpus / "notes" / "baking.md")  # new title
    rep = index(store, [corpus], emb)
    assert (rep.moved, rep.new, rep.removed) == (2, 1, 1)
    assert emb.calls["image"] == calls["image"] and emb.calls["docs"] == calls["docs"] + 1
    assert store.get_file(str(corpus / "pics" / "renamed-red.png")).status == "ok"
    assert store.get_file(str(corpus / "red.png")) is None
    hits = search(store, emb.embed_query("bicycle chain"), n=1)
    assert hits[0].path == str(corpus / "pics" / "bikes.txt")
    # A different budget means a new embedding.
    (corpus / "pics" / "renamed-red.png").rename(corpus / "red.png")
    rep = index(store, [corpus], emb, image_budget=280)
    assert rep.moved == 0


def test_nested_root_keeps_files_the_parent_ignores(tmp_path: Path, fake_embedder) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".gitignore").write_text("private/\n")
    (repo / "private").mkdir()
    (repo / "private" / "a.md").write_text("private note")
    (repo / "b.md").write_text("public note")
    store = Store(tmp_path / "idx.sqlite")
    index(store, [repo / "private"], fake_embedder)
    rep = index(store, [repo], fake_embedder)
    assert rep.removed == 0 and rep.new == 1
    assert store.get_file(str(repo / "private" / "a.md")) is not None


def test_pdf_pages_without_words_are_skipped(tmp_path: Path, fake_embedder) -> None:
    from .conftest import write_pdf

    root = tmp_path / "c"
    root.mkdir()
    write_pdf(root / "report.pdf", [["Results", "The pilot reduced waiting times."], ["5"]])
    write_pdf(root / "scan.pdf", [["12"], [""]])
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [root], fake_embedder)
    assert rep.new == 1 and rep.skipped == 1
    assert store.counts()["pdf"]["chunks"] == 1
    assert store.problem_files() == [
        (str(root / "scan.pdf"), "skipped", "no text layer (scanned PDF?)")
    ]


def test_image_too_large_is_skipped(tmp_path: Path, corpus: Path, fake_embedder, monkeypatch):
    from anygrep import embedder

    monkeypatch.setattr(embedder, "MAX_IMAGE_PIXELS", 10_000)
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [corpus], fake_embedder)
    assert rep.skipped == 1 and rep.errors == 0  # red.png (200x150); blue.jpg is a JPEG
    problems = {Path(p).name: (s, n) for p, s, n in store.problem_files()}
    assert problems["red.png"][0] == "skipped" and "too large" in problems["red.png"][1]


def test_memory_error_on_one_file_is_recorded(tmp_path: Path, corpus: Path, monkeypatch) -> None:
    from anygrep import indexer

    class Hungry(FakeEmbedder):
        def embed_image(self, image):
            if image.getpixel((0, 0))[0] > 200:  # red.png
                raise MemoryError
            return super().embed_image(image)

    real = indexer.load_image

    def load(path):
        if Path(path).name == "blue.jpg":
            raise MemoryError("decode")
        return real(path)

    monkeypatch.setattr(indexer, "load_image", load)
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [corpus], Hungry())
    assert rep.errors == 2 and rep.new == 3
    notes = {Path(p).name: n for p, _s, n in store.problem_files()}
    assert notes["red.png"].startswith("EmbedError: MemoryError")
    assert notes["blue.jpg"].startswith("MemoryError")
    # Both depend on the machine, not on the file: tried again next time.
    monkeypatch.setattr(indexer, "load_image", real)
    rep = index(store, [corpus], FakeEmbedder())
    assert rep.updated == 2 and rep.errors == 0


def test_large_file_is_committed_in_slices_and_continued(tmp_path: Path, monkeypatch) -> None:
    from anygrep import indexer

    monkeypatch.setattr(indexer, "LARGE_FILE_CHUNKS", 4)
    monkeypatch.setattr(indexer, "TEXT_BATCH_CHUNKS", 2)
    root = tmp_path / "c"
    root.mkdir()
    (root / "big.txt").write_text(
        "\n".join(f"line {i} about bread and ovens" for i in range(40)), encoding="utf-8"
    )
    (root / "small.md").write_text("A short note about bicycles.\n", encoding="utf-8")
    opts = dict(chunk_tokens=12, chunk_overlap=0)
    store = Store(tmp_path / "idx.sqlite")
    emb = FakeEmbedder()
    emb.fail_after = 3  # three slices of big.txt, then Ctrl-C
    rep = index(store, [root], emb, **opts)
    assert rep.interrupted
    row = store.get_file(str(root / "big.txt"))
    assert row is not None and row.status == "partial"
    n_done = store.db.execute("SELECT COUNT(*) FROM chunks WHERE file_id=?", (row.id,)).fetchone()
    assert n_done[0] == 6
    q = emb.embed_query("bread ovens")
    assert all(Path(h.path).name != "big.txt" for h in search(store, q, n=10))

    emb2 = FakeEmbedder()
    rep = index(store, [root], emb2, **opts)
    row = store.get_file(str(root / "big.txt"))
    assert row is not None and row.status == "ok" and rep.errors == 0
    ords = [r[0] for r in store.db.execute("SELECT ord FROM chunks WHERE file_id=?", (row.id,))]
    assert sorted(ords) == list(range(len(ords))) and len(ords) > 6
    small = 1 if store.get_file(str(root / "small.md")).status == "ok" else 0
    assert emb2.calls["docs"] == len(ords) - 6 + (small and 1)  # only the rest is embedded
    assert Path(search(store, q, n=1)[0].path).name == "big.txt"


def test_failing_slice_marks_the_large_file(tmp_path: Path, monkeypatch) -> None:
    from anygrep import indexer

    monkeypatch.setattr(indexer, "LARGE_FILE_CHUNKS", 4)
    monkeypatch.setattr(indexer, "TEXT_BATCH_CHUNKS", 2)

    class Flaky(FakeEmbedder):
        def embed_documents(self, texts, titles):
            if self.calls["docs"] >= 4:
                raise RuntimeError("onnx")
            return super().embed_documents(texts, titles)

    root = tmp_path / "c"
    root.mkdir()
    (root / "big.txt").write_text("\n".join(f"line {i}" for i in range(60)), encoding="utf-8")
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [root], Flaky(), chunk_tokens=6, chunk_overlap=0)
    assert rep.errors == 1
    row = store.get_file(str(root / "big.txt"))
    assert row is not None and row.status == "error" and row.note.startswith("EmbedError")
    assert not store.has_chunks()


def _writes_outside_transactions(store: Store, fn) -> list[str]:
    outside: list[str] = []
    depth = [0]

    def trace(sql: str) -> None:
        s = sql.strip().upper()
        if s.startswith("BEGIN"):
            depth[0] += 1
        elif s.startswith(("COMMIT", "ROLLBACK")):
            depth[0] -= 1
        elif s.startswith("UPDATE") and not depth[0]:
            outside.append(sql)

    store.db.set_trace_callback(trace)
    try:
        fn()
    finally:
        store.db.set_trace_callback(None)
    return outside


def test_touched_and_moved_files_are_updated_in_one_transaction(
    tmp_path: Path, fake_embedder
) -> None:
    root = tmp_path / "c"
    (root / "a").mkdir(parents=True)
    for i in range(30):
        (root / "a" / f"n{i}.md").write_text(f"note {i} about things", encoding="utf-8")
    store = Store(tmp_path / "idx.sqlite")
    index(store, [root], fake_embedder)
    for i in range(15):  # touched: same content, newer mtime
        p = root / "a" / f"n{i}.md"
        st = p.stat()
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    (root / "b").mkdir()
    for i in range(15, 30):  # moved
        (root / "a" / f"n{i}.md").rename(root / "b" / f"n{i}.md")
    reps = []
    outside = _writes_outside_transactions(
        store, lambda: reps.append(index(store, [root], fake_embedder))
    )
    assert reps[0].moved == 15 and reps[0].unchanged == 15
    assert outside == []


def test_move_of_a_row_deleted_meanwhile_indexes_the_file(
    tmp_path: Path, fake_embedder, monkeypatch
) -> None:
    root = tmp_path / "c"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x.md").write_text("bread and ovens", encoding="utf-8")
    store = Store(tmp_path / "idx.sqlite")
    index(store, [root], fake_embedder)
    (root / "b").mkdir()
    (root / "a" / "x.md").rename(root / "b" / "x.md")
    # Another process removes the row between planning and the move.
    monkeypatch.setattr(store, "move_file", lambda *a: False)
    rep = index(store, [root], fake_embedder)
    assert rep.moved == 0 and rep.new == 1
    assert store.get_file(str(root / "b" / "x.md")) is not None
    assert store.get_file(str(root / "a" / "x.md")) is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_index_files_are_private(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    path = tmp_path / "data" / "idx.sqlite"
    store = Store(path)
    index(store, [corpus], fake_embedder)
    assert (path.parent.stat().st_mode & 0o777) == 0o700
    for s in ("", "-wal", "-shm"):
        p = path.with_name(path.name + s)
        if p.exists():
            assert p.stat().st_mode & 0o077 == 0, s
    store.close()
    path.chmod(0o644)  # made by an older version
    Store(path).close()
    assert path.stat().st_mode & 0o077 == 0


def test_other_sqlite_files_are_not_touched(tmp_path: Path) -> None:
    import sqlite3

    from anygrep.store import IndexMismatchError

    other = tmp_path / "app.db"
    db = sqlite3.connect(other)
    db.execute("CREATE TABLE notes(x)")
    db.commit()
    db.close()
    for create in (False, True):
        with pytest.raises(IndexMismatchError, match="not an anygrep index"):
            Store(other, create=create)
    db = sqlite3.connect(other)
    assert [r[0] for r in db.execute("SELECT name FROM sqlite_master")] == ["notes"]
    db.close()


def test_new_index_settings_are_written_together(tmp_path: Path, monkeypatch) -> None:
    calls = []
    real = Store.set_meta

    def crash_after_first(self, key, value):
        calls.append(key)
        if len(calls) == 2:
            raise KeyboardInterrupt  # killed while creating the index
        real(self, key, value)

    monkeypatch.setattr(Store, "set_meta", crash_after_first)
    with pytest.raises(KeyboardInterrupt):
        Store(tmp_path / "idx.sqlite")
    monkeypatch.setattr(Store, "set_meta", real)
    st = Store(tmp_path / "idx.sqlite")  # no half-made index to reject
    assert st.get_meta("model_revision") and st.get_meta("schema_version")
    st.close()


def test_full_disk_error_is_not_hidden(tmp_path: Path) -> None:
    import sqlite3

    st = Store(tmp_path / "idx.sqlite")
    (pages,) = st.db.execute("PRAGMA page_count").fetchone()
    st.db.execute(f"PRAGMA max_page_count={pages + 2}")
    with pytest.raises(sqlite3.OperationalError, match="full"), st.transaction():
        for i in range(200):
            st.set_meta(f"k{i}", "x" * 100_000)
    assert not st.db.in_transaction
    st.close()
