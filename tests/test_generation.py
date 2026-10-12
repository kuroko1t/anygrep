"""Processing generations: files processed by an older anygrep are processed
again by the next update, without a reset (config.PROCESSING_GENERATION)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import anygrep
from anygrep import cli, config, embedder
from anygrep.indexer import IndexOptions, record_generation, run_index
from anygrep.store import Store

from .conftest import FakeEmbedder


def index(store: Store, roots, emb: FakeEmbedder, **kw):
    return run_index(store, list(roots), IndexOptions(quiet=True, **kw), lambda vision: emb)


def bump(monkeypatch, *kinds: str) -> None:
    """A new anygrep whose generation 2 changed how `kinds` are processed."""
    monkeypatch.setattr(config, "PROCESSING_GENERATION", 2)
    monkeypatch.setattr(config, "GENERATION_CHANGES", {2: frozenset(kinds)})


def test_new_rows_record_the_generation(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    gens = {r[0] for r in store.db.execute("SELECT gen FROM files")}
    assert gens == {config.PROCESSING_GENERATION}
    assert store.get_meta("processing_generation") == str(config.PROCESSING_GENERATION)
    assert store.count_stale() == 0


def test_update_processes_only_the_changed_kinds_again(
    tmp_path: Path, corpus: Path, fake_embedder, monkeypatch
) -> None:
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    calls = dict(fake_embedder.calls)

    bump(monkeypatch, "text")
    assert store.count_stale() == 2  # the two notes
    rep = index(store, [corpus], fake_embedder)
    assert (rep.outdated, rep.updated, rep.unchanged, rep.new) == (2, 2, 3, 0)
    # Only the notes were embedded again: no image, no PDF.
    assert fake_embedder.calls["image"] == calls["image"]
    assert fake_embedder.calls["docs"] == calls["docs"] + 2
    assert store.count_stale() == 0
    record_generation(store)
    assert store.get_meta("processing_generation") == "2"

    # Nothing left to do on the next run.
    rep = index(store, [corpus], fake_embedder)
    assert (rep.outdated, rep.updated, rep.unchanged) == (0, 0, 5)


def test_interrupted_reprocessing_resumes(
    tmp_path: Path, corpus: Path, fake_embedder, monkeypatch
) -> None:
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    bump(monkeypatch, "image")
    fake_embedder.fail_after = 1  # the first image, then Ctrl-C
    rep = index(store, [corpus], fake_embedder)
    assert rep.interrupted
    assert store.count_stale() == 1
    fake_embedder.fail_after = None
    rep = index(store, [corpus], fake_embedder)
    assert rep.outdated == 1 and store.count_stale() == 0


def test_identical_images_of_an_older_generation_are_not_copied(
    tmp_path: Path, corpus: Path, fake_embedder, monkeypatch
) -> None:
    store = Store(tmp_path / "idx.sqlite")
    index(store, [corpus], fake_embedder)
    bump(monkeypatch, "image")
    # A copy of an outdated image must be embedded, not given the old vector.
    (corpus / "copy.png").write_bytes((corpus / "red.png").read_bytes())
    before = fake_embedder.calls["image"]
    rep = index(store, [corpus], fake_embedder)
    assert rep.reused == 1  # the copy took the vector embedded in this run
    assert fake_embedder.calls["image"] == before + 2


def test_index_from_before_generations_is_generation_one(
    tmp_path: Path, corpus: Path, fake_embedder
) -> None:
    """An index made before rows recorded a generation: nothing to redo
    until a later generation changes something."""
    path = tmp_path / "idx.sqlite"
    store = Store(path)
    index(store, [corpus], fake_embedder)
    store.close()
    db = sqlite3.connect(path)
    db.execute("ALTER TABLE files DROP COLUMN gen")
    db.execute("DELETE FROM meta WHERE key='processing_generation'")
    db.commit()
    db.close()
    ro = Store(path, readonly=True)
    assert ro.count_stale() == 0
    ro.close()
    store = Store(path)
    assert store.count_stale() == 0
    rep = index(store, [corpus], fake_embedder)
    assert rep.unchanged == 5


def test_search_notes_outdated_files(env, corpus: Path, monkeypatch, capsys) -> None:
    assert cli.main(["index", "-q", str(corpus)]) == 0
    capsys.readouterr()
    bump(monkeypatch, "text", "pdf")

    # A read-only search still searches, and says once that an update is due.
    code = cli.main(["--no-index", "bread", str(corpus)])
    err = capsys.readouterr().err
    assert code == 0
    assert err.count("indexed by an older version of anygrep") == 1
    assert "3 files here were" in err

    # status shows them too.
    assert cli.main(["status"]) == 0
    assert "Outdated: 3 files" in capsys.readouterr().out

    # A search that may write processes them again first (the scope only).
    assert cli.main(["bread", str(corpus / "notes")]) == 0
    err = capsys.readouterr().err
    assert "older version" not in err
    st = Store(cli.default_index_path(), create=False)
    try:
        assert st.count_stale([str(corpus / "notes")]) == 0
        assert st.count_stale() == 1  # the PDF outside the scope
        assert st.get_meta("processing_generation") == "1"
    finally:
        st.close()

    # `anygrep index` finishes the rest and records the generation.
    assert cli.main(["index", str(corpus)]) == 0
    err = capsys.readouterr().err
    assert "1 file was indexed by an older version of anygrep" in err
    st = Store(cli.default_index_path(), create=False)
    try:
        assert st.count_stale() == 0
        assert st.get_meta("processing_generation") == "2"
    finally:
        st.close()


@pytest.fixture
def api_index(tmp_path: Path, monkeypatch, env: FakeEmbedder):
    monkeypatch.setattr(embedder, "embedder_factory", lambda threads, quiet=True: lambda v: env)
    with anygrep.Index(tmp_path / "api.sqlite", threads=1) as i:
        yield i


def test_api_update_pending(api_index: anygrep.Index, corpus: Path, monkeypatch) -> None:
    api_index.update(corpus)
    assert api_index.update_pending is False
    bump(monkeypatch, "pdf")
    assert api_index.update_pending is True
    assert api_index.search("router manual", weak=True)  # still searchable
    api_index.update()
    assert api_index.update_pending is False
