"""Exact matches: files whose text or name contains the query come first."""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path

import pytest

import anygrep
from anygrep import cli, embedder, store, thresholds
from anygrep import search as search_mod
from anygrep.indexer import IndexOptions, run_index
from anygrep.search import scope_matcher, search
from anygrep.store import Store

from .conftest import FakeEmbedder, jsonl, write_pdf

pytestmark = pytest.mark.skipif(
    not store.fts_available(), reason="this SQLite has no FTS5 trigram tokenizer"
)

FILLER = " ".join(f"word{i}" for i in range(60))


@pytest.fixture
def cutoff(monkeypatch):
    for pair in ("text->text", "text->pdf"):
        monkeypatch.setitem(thresholds.MIN_SCORES, pair, {0: 0.3})
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.3})


@pytest.fixture
def docs(corpus: Path) -> Path:
    """invoice.md holds the reference among many other words (a weak match
    by meaning); close.md shares its words but not the string."""
    (corpus / "notes" / "invoice.md").write_text(
        f"Payment reference ABC-77812 for the garden shed.\n{FILLER}\n", encoding="utf-8"
    )
    (corpus / "notes" / "close.md").write_text("ABC 77812 ABC 77812\n", encoding="utf-8")
    (corpus / "Q3-budget-draft.txt").write_text("Numbers for the coming months.\n")
    write_pdf(corpus / "contract.pdf", [["Lease terms"], ["Contract number LX-4471 applies."]])
    return corpus


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def index(store_: Store, root: Path, emb: FakeEmbedder) -> None:
    run_index(store_, [root], IndexOptions(quiet=True), lambda vision: emb)


def check(st: Store) -> None:
    """The FTS index matches the chunk text exactly."""
    t = store.FTS_TABLE
    st.db.execute(f"INSERT INTO {t}({t}, rank) VALUES('integrity-check', 1)")


def query(st: Store, emb: FakeEmbedder, q: str, **kw):
    kw.setdefault("hide_weak", True)
    return search(st, emb.embed_query(q), exact=q, **kw)


def test_exact_text_match_comes_first_even_when_weak(
    tmp_path: Path, docs: Path, fake_embedder, cutoff
) -> None:
    st = Store(tmp_path / "idx.sqlite")
    index(st, docs, fake_embedder)
    hits = query(st, fake_embedder, "ABC-77812")
    assert [Path(h.path).name for h in hits[:2]] == ["invoice.md", "close.md"]
    assert hits[0].exact and not hits[0].weak and hits[0].score < 0.3
    assert not hits[1].exact
    assert "ABC-77812" in (hits[0].text or "")
    # Without the exact string, invoice.md is below the cutoff.
    plain = search(st, fake_embedder.embed_query("ABC-77812"), hide_weak=True)
    assert "invoice.md" not in [Path(h.path).name for h in plain]


def test_case_pdf_name_and_false_positives(
    tmp_path: Path, docs: Path, fake_embedder, cutoff
) -> None:
    st = Store(tmp_path / "idx.sqlite")
    index(st, docs, fake_embedder)

    def exact(q: str, **kw) -> list[str]:
        return [Path(h.path).name for h in query(st, fake_embedder, q, **kw) if h.exact]

    assert exact("abc-77812") == ["invoice.md"]  # case is ignored
    hits = query(st, fake_embedder, "lx-4471")
    assert hits[0].exact and hits[0].kind == "pdf" and hits[0].page == 2
    assert exact("budget-draft") == ["Q3-budget-draft.txt"]  # file name
    # The file name only, not the folders above it.
    assert exact("notes/inv", prefixes=[str(docs)]) == []
    assert exact("notes") == []
    # SQLite LIKE wildcards in the query are not wildcards here.
    assert exact("ABC_77812") == [] and exact("ABC%7812") == []
    assert exact("AB") == []  # under 3 characters: meaning only
    # An image query never matches exactly.
    hits = search(st, fake_embedder.embed_query("red"), query_kind="image", exact="red.png")
    assert not any(h.exact for h in hits)


def test_exact_respects_scope_type_and_count(
    tmp_path: Path, docs: Path, fake_embedder, cutoff
) -> None:
    for i in range(3):
        (docs / f"ref{i}.txt").write_text(f"code ZQX-{i}00 here\n")
    st = Store(tmp_path / "idx.sqlite")
    index(st, docs, fake_embedder)
    hits = query(st, fake_embedder, "ZQX-", n=2)
    assert len(hits) == 2 and all(h.exact for h in hits)
    hits = query(st, fake_embedder, "ABC-77812", kinds=["pdf"])
    assert not any(h.exact for h in hits)
    scope = scope_matcher([str(docs)], ["!notes/*"])
    hits = query(st, fake_embedder, "ABC-77812", scope=scope)
    assert not any(h.exact for h in hits)
    hits = query(st, fake_embedder, "ZQX-", per_file=False, n=10)
    assert sum(h.exact for h in hits) == 3


def test_fts_follows_changes(tmp_path: Path, docs: Path, fake_embedder, cutoff) -> None:
    st = Store(tmp_path / "idx.sqlite")
    index(st, docs, fake_embedder)
    assert st.exact_state() == "ready"
    (docs / "notes" / "invoice.md").write_text("Now the reference is DEF-1234.\n")
    (docs / "contract.pdf").unlink()
    index(st, docs, fake_embedder)
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))
    assert not any(h.exact for h in query(st, fake_embedder, "LX-4471"))
    assert query(st, fake_embedder, "def-1234")[0].exact
    check(st)
    n_fts = st.db.execute(f"SELECT COUNT(*) FROM {store.FTS_TABLE}_docsize").fetchone()[0]
    n_text = st.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert n_fts == n_text


def test_index_from_before_exact_matching(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    """An index without the FTS table: searched by reading the chunk text
    until an update fills the table in (resumably)."""
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    st.close()
    db = sqlite3.connect(path)
    db.execute(f"DROP TABLE {store.FTS_TABLE}")
    db.commit()
    db.close()

    ro = Store(path, readonly=True)
    assert query(ro, fake_embedder, "ABC-77812")[0].exact
    ro.close()

    monkeypatch.setattr(store, "FTS_BATCH", 2)
    st = Store(path)  # a writer creates the table; the chunks wait for an update
    assert st.exact_state() == "building" and st.fts_gap() is not None
    assert query(st, fake_embedder, "ABC-77812")[0].exact
    assert st.fill_fts(max_seconds=0) is False  # out of time: nothing done yet
    st.fill_fts()
    assert st.exact_state() == "ready" and st.fts_gap() is None
    assert query(st, fake_embedder, "ABC-77812")[0].exact
    check(st)


def test_changes_while_the_table_is_filled(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    st.close()
    db = sqlite3.connect(path)
    db.execute(f"DROP TABLE {store.FTS_TABLE}")
    db.commit()
    db.close()
    monkeypatch.setattr(store, "FTS_BATCH", 2)
    st = Store(path)
    clock = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(store.time, "monotonic", lambda: next(clock, 100.0))
    assert st.fill_fts(max_seconds=1) is False  # one batch of two chunks
    monkeypatch.undo()
    monkeypatch.setattr(store, "FTS_BATCH", 2)
    pos, end = st.fts_gap()
    assert 0 < pos < end
    # Change files before, inside and after the gap, then fill it.
    (docs / "notes" / "invoice.md").write_text("Reference GHI-5550 now.\n")
    (docs / "notes" / "new.md").write_text("Another one: JKL-9090.\n")
    index(st, docs, fake_embedder)
    st.fill_fts()
    check(st)
    for q in ("GHI-5550", "JKL-9090", "LX-4471"):
        assert query(st, fake_embedder, q)[0].exact, q
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))


def test_without_fts5_searches_by_meaning_only(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    st.close()
    monkeypatch.setattr(store, "_fts_ok", False)
    st = Store(path)
    assert st.exact_state() == "unavailable"
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))
    (docs / "notes" / "invoice.md").write_text("Reference MNO-3131 now.\n")
    index(st, docs, fake_embedder)  # writes without the FTS table
    st.close()
    monkeypatch.setattr(store, "_fts_ok", True)
    st = Store(path)  # rebuilt from the chunks by the next writer
    assert st.exact_state() == "building"
    st.fill_fts()
    assert query(st, fake_embedder, "MNO-3131")[0].exact
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))
    check(st)


def test_cli_marks_exact_matches(env, docs: Path, cutoff, capsys, tty, monkeypatch) -> None:
    code, out, _ = run(capsys, "ABC-77812", str(docs), "--json")
    rows = jsonl(out)
    assert code == 0 and rows[0]["exact"] is True and rows[0]["path"].endswith("invoice.md")
    assert all(r["exact"] is False for r in rows[1:])
    code, out, _ = run(capsys, "ABC-77812", str(docs), "-l")
    assert out.splitlines()[0].endswith("invoice.md")
    tty()
    code, out, _ = run(capsys, "ABC-77812", str(docs))
    first = out.splitlines()[0]
    assert "invoice.md" in first and first.endswith("(exact)")
    monkeypatch.setattr(store, "_fts_ok", False)
    code, out, _ = run(capsys, "status")
    assert "Exact matches: off" in out


def test_api_hit_exact(tmp_path: Path, docs: Path, env, cutoff, monkeypatch) -> None:
    monkeypatch.setattr(embedder, "embedder_factory", lambda threads, quiet=True: lambda v: env)
    with anygrep.Index(tmp_path / "api.sqlite", threads=1) as idx:
        idx.update(docs)
        hits = idx.search("ABC-77812")
        assert hits[0].exact and hits[0].path.endswith("invoice.md") and not hits[0].weak
        assert not idx.search("AB", weak=True)[0].exact


def test_writer_opened_before_the_table_was_made(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    """A writer that opened an index before another process created the FTS
    table still keeps it up to date."""
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    st.close()
    db = sqlite3.connect(path)
    db.execute(f"DROP TABLE {store.FTS_TABLE}")
    db.commit()
    db.close()
    monkeypatch.setattr(store, "_fts_ok", False)
    early = Store(path, readonly=True)  # opened without the table
    early.close()
    early = Store(path)
    assert not early.fts
    monkeypatch.setattr(store, "_fts_ok", True)
    early.db.execute("DELETE FROM meta WHERE key='fts_reset'")
    other = Store(path)  # makes the table
    (docs / "notes" / "late.md").write_text("Late reference PQR-6060.\n")
    index(early, docs, fake_embedder)
    other.fill_fts()
    check(other)
    assert query(other, fake_embedder, "PQR-6060")[0].exact


def test_case_and_width_of_any_script(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    """Case is ignored for every script, and text is compared in NFKC form:
    decomposed accents and full-width letters and digits match too."""
    (corpus / "ru.txt").write_text("Москва Россия. Große Äpfel und Öl.\n", encoding="utf-8")
    nfd = unicodedata.normalize("NFD", "café in Zürich")
    (corpus / "nfd.txt").write_text(nfd + "\n", encoding="utf-8")
    (corpus / "full.txt").write_text("Ｗｉｆｉ設定 ＡＢＣ１２３\n", encoding="utf-8")
    st = Store(tmp_path / "idx.sqlite")
    index(st, corpus, fake_embedder)

    def exact(q: str) -> list[str]:
        return sorted(Path(h.path).name for h in query(st, fake_embedder, q) if h.exact)

    for q in ("москва россия", "МОСКВА РОССИЯ", "Москва"):
        assert exact(q) == ["ru.txt"], q
    assert exact("große äpfel") == exact("GROSSE ÄPFEL") == ["ru.txt"]
    assert exact("zürich") == exact("café in") == ["nfd.txt"]
    for q in ("wifi設定", "ABC123", "abc１２３", "ＡＢＣ１２３"):
        assert exact(q) == ["full.txt"], q
    check(st)


def test_short_words_and_common_strings(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    """A query of 3 or 4 ASCII letters matches whole words only, and a
    string found in many of the files searched gives no exact matches."""
    (corpus / "a.txt").write_text("Do not run this script twice.\n")
    (corpus / "b.txt").write_text("It truncates the log and prunes old files.\n")
    (corpus / "c.txt").write_text("Set up the pipeline first.\n")
    (corpus / "d.txt").write_text("Install it with pip, then restart.\n")
    for i in range(12):
        (corpus / f"common{i}.txt").write_text(f"Entry {i} of the list: zebra{i}.\n")
    st = Store(tmp_path / "idx.sqlite")
    index(st, corpus, fake_embedder)

    def exact(q: str) -> list[str]:
        hits = query(st, fake_embedder, q, n=30, hide_weak=False)
        return sorted(Path(h.path).name for h in hits if h.exact)

    assert exact("run") == ["a.txt"]
    assert exact("pip") == ["d.txt"]
    assert exact("pipe") == []
    assert exact("pipeline") == ["c.txt"]  # longer queries match inside words
    assert exact("trunc") == ["b.txt"]
    # "the" is in 14 files, more than the limit of 10 for 17 files.
    assert exact("the") == []
    assert exact("zebra") == []  # 12 files, also more than the limit
    assert search_mod.EXACT_MAX_FILES < 12
    assert exact("zebra1") == ["common1.txt", "common10.txt", "common11.txt"]


def test_query_syntax_is_literal(tmp_path: Path, corpus: Path, fake_embedder) -> None:
    (corpus / "q.txt").write_text('He said "stop" (NOT now) AND left; 50% of a_b OR c*d; x"y\n')
    st = Store(tmp_path / "idx.sqlite")
    index(st, corpus, fake_embedder)
    for q in ('"stop"', "(NOT now)", "AND left", "50%", "a_b OR", "c*d", 'x"y', "now) AND"):
        hits = query(st, fake_embedder, q)
        assert hits and hits[0].exact and hits[0].path.endswith("q.txt"), q
    for q in ("a%b", "50_", "c?d", '"AND"', "NOT stop"):
        assert not any(h.exact for h in query(st, fake_embedder, q)), q


def test_very_long_query(tmp_path: Path, docs: Path, fake_embedder, cutoff) -> None:
    """A query of any length is looked up without an error, also where the
    chunk text is read directly (a read-only index made before exact
    matching)."""
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    long = "x" * 60000
    assert not any(h.exact for h in query(st, fake_embedder, long))
    text = "Payment reference ABC-77812 for the garden shed."
    assert query(st, fake_embedder, text)[0].exact
    st.close()
    drop_fts(path)
    ro = Store(path, readonly=True)
    assert not ro.fts
    assert not any(h.exact for h in query(ro, fake_embedder, long))
    assert query(ro, fake_embedder, "ABC-77812")[0].exact


def test_lookup_runs_once_per_state(tmp_path: Path, docs: Path, fake_embedder, monkeypatch):
    st = Store(tmp_path / "idx.sqlite")
    index(st, docs, fake_embedder)
    calls = []
    real = st._exact_lookup
    monkeypatch.setattr(st, "_exact_lookup", lambda *a: calls.append(a) or real(*a))
    query(st, fake_embedder, "ABC-77812")
    query(st, fake_embedder, "ABC-77812", hide_weak=False)
    assert len(calls) == 1
    (docs / "notes" / "invoice.md").write_text("Now the reference is DEF-1234.\n")
    index(st, docs, fake_embedder)
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))
    assert len(calls) == 2


def test_schema_moves_to_two(tmp_path: Path, docs: Path, fake_embedder) -> None:
    """An index of schema 1 (anygrep 0.2) is read as it is and moved to
    schema 2 by a writer, which anygrep 0.2 then refuses: it would change
    the chunks without the FTS table."""
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    st.close()
    drop_fts(path)
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
    db.execute("DELETE FROM meta WHERE key='fts_format'")
    db.commit()
    db.close()
    ro = Store(path, readonly=True)
    assert ro.get_meta("schema_version") == "1"
    ro.close()
    st = Store(path)
    assert st.get_meta("schema_version") == str(store.SCHEMA_VERSION) == "2"


def test_damaged_fts_table_is_rebuilt(tmp_path: Path, docs: Path, fake_embedder, cutoff):
    """A write that finds the FTS table damaged still succeeds, lookups read
    the chunk text until then, and the next update rebuilds the table."""
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    st.close()
    db = sqlite3.connect(path)
    db.execute(f"UPDATE {store.FTS_TABLE}_data SET block=x'ffffffff'")
    db.commit()
    db.close()
    st = Store(path)
    (docs / "notes" / "invoice.md").write_text("Reference GHI-5550 now.\n")
    index(st, docs, fake_embedder)
    assert st.get_meta("fts_reset") == "1" and st.exact_state() == "building"
    assert query(st, fake_embedder, "GHI-5550")[0].exact
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))
    st.fill_fts()
    assert st.exact_state() == "ready"
    check(st)
    assert query(st, fake_embedder, "GHI-5550")[0].exact
    assert query(st, fake_embedder, "LX-4471")[0].exact


def drop_fts(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute(f"DROP TABLE {store.FTS_TABLE}")
    db.commit()
    db.close()
