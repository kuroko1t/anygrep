"""Exact matches: files whose text or name contains the query come first."""

from __future__ import annotations

import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
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
    assert st.fill_fts(max_seconds=-1) is False  # out of time: nothing done yet
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
    code, out, _ = run(capsys, "-h")
    assert code == 0 and "marked (exact)" in out


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


def test_ids_are_always_matched_common_words_are_not(
    tmp_path: Path, corpus: Path, fake_embedder
) -> None:
    """A query with a digit or a symbol is matched however many files have
    it; plain words are not when they are in too many of the files searched,
    which in a small scope means most of them."""
    shared = [f"inv{i}.txt" for i in range(12)]
    for name in shared:
        (corpus / name).write_text(f"Invoice INV-2026-0412 copy {name}.\n")
    for i in range(30):
        (corpus / f"other{i}.txt").write_text(f"Unrelated note {i} about zebras.\n")
    small = corpus / "small"
    small.mkdir()
    for i in range(5):
        (small / f"s{i}.txt").write_text(f"The note {i} with code Q-77{i} and the end.\n")
    (small / "s0.txt").write_text("The kayak and the paddle, code Q-770.\n")
    (small / "s1.txt").write_text("The kayak trip and the lake, code Q-771.\n")
    words = corpus / "words"
    words.mkdir()
    for i in range(5):
        (words / f"w{i}.txt").write_text(
            f"नोट ที่นี่ İstanbul follow-up e-mail, entry {i}.\n", encoding="utf-8"
        )
    st = Store(tmp_path / "idx.sqlite")
    index(st, corpus, fake_embedder)

    def exact(q: str, *where: Path) -> list[str]:
        scope = scope_matcher([str(p) for p in where], []) if where else None
        hits = query(st, fake_embedder, q, n=100, hide_weak=False, scope=scope)
        return sorted(Path(h.path).name for h in hits if h.exact)

    # The same ID in 12 related files (more than 10): all of them.
    assert exact("INV-2026-0412") == sorted(shared)
    assert exact("inv-2026") == sorted(shared)
    # A plain word in 30 files: none.
    assert exact("zebras") == []
    # "the" in all 5 files of a small folder: none; an ID there: its file.
    assert exact("the", small) == []
    assert exact("Q-773", small) == ["s3.txt"]
    # A plain word in 2 of 5 files is a clue, in 3 of 5 it is not.
    assert exact("kayak", small) == ["s0.txt", "s1.txt"]
    assert exact("note", small) == []
    # One file searched: its words count, also common ones.
    assert exact("the", small / "s2.txt") == ["s2.txt"]
    assert exact("the", small / "s2.txt", small / "s3.txt") == ["s2.txt", "s3.txt"]
    # Words with combining marks (Hindi, Thai, Turkish "İ" once folded) or
    # a hyphen inside are plain words too: in every file of "words", none.
    for q in ("नोट", "ที่นี่", "İstanbul", "follow-up", "e-mail"):
        assert exact(q, words) == [], q
        assert exact(q, words / "w1.txt") == ["w1.txt"], q
    for q in ("don't stop", "e-tax", "follow-up", "नोट", "مَرْحَبًا", "i\u0307stanbul"):
        assert search_mod.is_plain_words(q), q
    for q in ("v2.14.3", "report.pdf", "6,850 kwh", "a_b", "q-7", "-x", "a--b"):
        assert not search_mod.is_plain_words(q), q


WORD_CASES = {
    # text or file name stem: whether "run" matches it as a whole word
    "run": True,
    "runé": False,
    "érun": False,
    "run2": False,
    "truncates": False,
    "run_script": True,
    "run-script": True,
    "(run)": True,
    "runを実行": True,
    "実行run": True,
    "ＲＵＮする": True,  # full width, NFKC folds it to "run"
    "ｒｕｎｅ": False,  # full-width "rune"
    "run\u0301": False,  # n with a combining accent (composed to "ń")
}


def test_word_pattern_rules() -> None:
    pat = store.word_pattern("run")
    for text, want in WORD_CASES.items():
        assert (pat.search(store.fold(text)) is not None) is want, text


@pytest.mark.parametrize("where", ["text", "name"])
def test_short_words_whole_word_in_text_and_names(
    tmp_path: Path, corpus: Path, fake_embedder, where: str
) -> None:
    """The same whole-word rule applies to the text and to file names."""
    want = set()
    for i, (stem, ok) in enumerate(WORD_CASES.items()):
        name = f"t{i}.txt" if where == "text" else f"{stem}.txt"
        body = f"Line {stem} here.\n" if where == "text" else f"Other words {i}.\n"
        (corpus / name).write_text(body, encoding="utf-8")
        if ok:
            want.add(unicodedata.normalize("NFC", name))
    for i in range(10):
        (corpus / f"filler{i}.txt").write_text(f"Nothing to see {i}.\n")
    st = Store(tmp_path / "idx.sqlite")
    index(st, corpus, fake_embedder)
    hits = query(st, fake_embedder, "run", n=100, hide_weak=False)
    got = {unicodedata.normalize("NFC", Path(h.path).name) for h in hits if h.exact}
    assert got == want


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


def partly_filled(path: Path, docs: Path, emb: FakeEmbedder, monkeypatch) -> Store:
    """An index whose FTS table holds only the first batch of chunks (two
    images and two texts)."""
    st = Store(path)
    index(st, docs, emb)
    st.close()
    drop_fts(path)
    monkeypatch.setattr(store, "FTS_BATCH", 4)
    st = Store(path)
    clock = iter([0.0, 0.0, 100.0])
    with monkeypatch.context() as m:
        m.setattr(store.time, "monotonic", lambda: next(clock, 100.0))
        assert st.fill_fts(max_seconds=1) is False
    pos, end = st.fts_gap()
    assert pos == 4 < end
    return st


def damage_fts(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute(f"UPDATE {store.FTS_TABLE}_data SET block=x'ffffffff' WHERE id>1")
    db.commit()
    db.close()


def test_damage_found_while_filling_rebuilds_the_table(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    """A partly filled FTS table that is damaged: the next fill makes it
    again from the chunks instead of failing on every update."""
    path = tmp_path / "idx.sqlite"
    st = partly_filled(path, docs, fake_embedder, monkeypatch)
    chunks = st.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    damage_fts(path)
    assert st.fill_fts() is True
    assert st.exact_state() == "ready" and st.get_meta("fts_reset") is None
    assert st.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == chunks
    check(st)
    assert query(st, fake_embedder, "LX-4471")[0].exact
    assert query(st, fake_embedder, "ABC-77812")[0].exact
    # Updates go on, and the table stays right.
    (docs / "notes" / "invoice.md").write_text("Reference GHI-5550 now.\n")
    index(st, docs, fake_embedder)
    assert st.fill_fts() is True
    check(st)
    assert query(st, fake_embedder, "GHI-5550")[0].exact
    assert not any(h.exact for h in query(st, fake_embedder, "ABC-77812"))


def test_update_after_damage_while_filling(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    path = tmp_path / "idx.sqlite"
    st = partly_filled(path, docs, fake_embedder, monkeypatch)
    st.close()
    damage_fts(path)
    st = Store(path)
    (docs / "notes" / "new.md").write_text("Another one: JKL-9090.\n")
    from anygrep.indexer import update_index

    update_index(st, [docs], IndexOptions(quiet=True), lambda vision: fake_embedder)
    st.fill_fts()
    assert st.exact_state() == "ready"
    check(st)
    for q in ("JKL-9090", "LX-4471", "ABC-77812"):
        assert query(st, fake_embedder, q)[0].exact, q


class FailingFTSInserts:
    """A connection whose inserts into the FTS table fail as FTS5 reports
    some damage."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    def execute(self, sql: str, *args):
        if sql.startswith(f"INSERT INTO {store.FTS_TABLE}(rowid"):
            raise sqlite3.IntegrityError("constraint failed")
        return self._db.execute(sql, *args)

    def __getattr__(self, name: str):
        return getattr(self._db, name)


def test_failed_batch_is_not_counted_as_filled(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, monkeypatch
) -> None:
    """When even the new table fails, the fill reports that it is not done
    and keeps the position of the batch that failed."""
    path = tmp_path / "idx.sqlite"
    st = partly_filled(path, docs, fake_embedder, monkeypatch)
    real = st.db
    st.db = FailingFTSInserts(real)
    assert st.fill_fts() is False
    assert st.get_meta("fts_reset") == "1" and st.exact_state() == "building"
    assert st.fts_gap()[0] == 0  # nothing counted as filled
    assert query(st, fake_embedder, "LX-4471")[0].exact  # read from the chunks
    st.db = real
    assert st.fill_fts() is True
    check(st)


def test_constraint_errors_of_other_tables_are_not_damage(tmp_path: Path) -> None:
    db = sqlite3.connect(tmp_path / "t.sqlite")
    db.execute("CREATE TABLE t(id INTEGER PRIMARY KEY)")
    db.execute("INSERT INTO t VALUES(1)")
    with pytest.raises(sqlite3.IntegrityError) as e:
        db.execute("INSERT INTO t VALUES(1)")
    assert not store._corrupt(e.value)
    assert store._corrupt(e.value, fts_write=True)  # only for FTS writes
    assert not store._corrupt(sqlite3.OperationalError("database is locked"), fts_write=True)
    db.close()


def built(path: Path, docs: Path, emb: FakeEmbedder, damage: str) -> None:
    """A complete index whose FTS tables are then changed by `damage`."""
    st = Store(path)
    index(st, docs, emb)
    st.close()
    db = sqlite3.connect(path)
    db.execute(damage.format(t=store.FTS_TABLE))
    db.commit()
    db.close()


def update(st: Store, docs: Path, emb: FakeEmbedder) -> None:
    from anygrep.indexer import update_index

    update_index(st, [docs], IndexOptions(quiet=True), lambda vision: emb)


@pytest.mark.parametrize("block", ["x'ffffffff'", "x'00'", "x''"])
def test_damaged_structure_record_found_without_changes(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, block: str
) -> None:
    """A damaged structure record can make every lookup find nothing, with
    no error: the next update notices it and makes the table again."""
    path = tmp_path / "idx.sqlite"
    built(path, docs, fake_embedder, f"UPDATE {{t}}_data SET block={block} WHERE id=10")
    st = Store(path, readonly=True)
    query(st, fake_embedder, "LX-4471")  # no error
    st.close()
    st = Store(path)
    update(st, docs, fake_embedder)
    assert st.exact_state() == "ready" and st.get_meta("fts_reset") is None
    check(st)
    for q in ("LX-4471", "ABC-77812"):
        assert query(st, fake_embedder, q)[0].exact, q


@pytest.mark.parametrize(
    "damage",
    [
        "UPDATE {t}_data SET block=substr(block, 1, length(block) / 2) "
        "WHERE id=(SELECT max(id) FROM {t}_data)",
        "DELETE FROM {t}_data WHERE id>10",
    ],
)
def test_damage_found_by_a_lookup_is_recorded(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, damage: str
) -> None:
    path = tmp_path / "idx.sqlite"
    built(path, docs, fake_embedder, damage)
    ro = Store(path, readonly=True)
    assert query(ro, fake_embedder, "LX-4471")[0].exact  # read from the chunks
    ro.close()
    st = Store(path)
    assert query(st, fake_embedder, "LX-4471")[0].exact
    assert st.get_meta("fts_reset") == "1"
    assert st.fill_fts() is True
    assert st.exact_state() == "ready"
    check(st)
    assert query(st, fake_embedder, "ABC-77812")[0].exact


def test_damage_seen_only_by_a_reader_is_found_by_the_next_update(
    tmp_path: Path, docs: Path, fake_embedder, cutoff
) -> None:
    path = tmp_path / "idx.sqlite"
    built(path, docs, fake_embedder, "DELETE FROM {t}_data WHERE id>10")
    st = Store(path)
    update(st, docs, fake_embedder)
    assert st.exact_state() == "ready"
    check(st)


@pytest.mark.parametrize(
    "damage",
    [
        "DELETE FROM {t}_data WHERE id=10",
        "UPDATE {t}_config SET v=99 WHERE k='version'",
    ],
)
def test_table_that_cannot_be_opened_is_made_again(
    tmp_path: Path, docs: Path, fake_embedder, cutoff, damage: str
) -> None:
    """A missing structure record or broken settings: SQLite cannot even
    drop the table, yet searches work and updates make it again."""
    path = tmp_path / "idx.sqlite"
    built(path, docs, fake_embedder, damage)
    ro = Store(path, readonly=True)
    assert query(ro, fake_embedder, "LX-4471")[0].exact
    ro.close()
    st = Store(path)
    (docs / "notes" / "new.md").write_text("Another one: JKL-9090.\n")
    update(st, docs, fake_embedder)
    st.close()
    for _ in range(2):  # later runs open and update it too
        st = Store(path)
        update(st, docs, fake_embedder)
        assert st.exact_state() == "ready" and st.get_meta("fts_reset") is None
        check(st)
        for q in ("JKL-9090", "LX-4471", "ABC-77812"):
            assert query(st, fake_embedder, q)[0].exact, q
        st.close()
    db = sqlite3.connect(path)
    assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    db.close()


class DamageOnFirstWrite(FailingFTSInserts):
    """A connection whose first write to the FTS table fails as damage."""

    failed = False

    def execute(self, sql: str, *args):
        if not self.failed and sql.startswith(f"INSERT INTO {store.FTS_TABLE}("):
            self.failed = True
            raise sqlite3.DatabaseError("database disk image is malformed")
        return self._db.execute(sql, *args)


def test_one_update_completes_the_table_after_damage(
    tmp_path: Path, docs: Path, fake_embedder, cutoff
) -> None:
    """Damage found by a write during `anygrep index`: the same run makes
    the table again, so it is complete when the run ends."""
    path = tmp_path / "idx.sqlite"
    st = Store(path)
    index(st, docs, fake_embedder)
    real = st.db
    st.db = DamageOnFirstWrite(real)
    (docs / "notes" / "invoice.md").write_text("Reference ZZQ-31337 now.\n")
    update(st, docs, fake_embedder)
    assert st.db.failed
    st.db = real
    assert st.exact_state() == "ready" and st.fts_gap() is None
    check(st)
    assert query(st, fake_embedder, "ZZQ-31337")[0].exact


def test_settings_error_counts_as_damage() -> None:
    e = sqlite3.OperationalError("invalid fts5 file format (found 99, expected 4 or 5)")
    assert store._corrupt(e)
    assert not store._corrupt(sqlite3.OperationalError("no such table: x"))


def chunk(text: str) -> store.ChunkData:
    return store.ChunkData(0, np.ones(store.EMBED_DIM, np.float32), text)


def chunk_ids(st: Store) -> set[int]:
    return {r[0] for r in st.db.execute("SELECT id FROM chunks")}


def one_chunk(path: Path) -> None:
    st = Store(path)
    with st.transaction():
        st.put_file("/n/a.txt", "text", 1, 1, "s", None, "ok", None, [chunk("abc foo")])
    st.close()


def test_damage_found_in_a_snapshot_another_writer_moved_past(tmp_path: Path) -> None:
    """A lookup that finds the FTS index damaged inside a read snapshot, after
    another connection has committed, reads the chunk text instead of
    failing on a write the snapshot cannot make; the damage is recorded once
    the snapshot ends, and fill_fts() makes the table again."""
    path = tmp_path / "idx.sqlite"
    one_chunk(path)
    db = sqlite3.connect(path)
    db.execute(f"DELETE FROM {store.FTS_TABLE}_data WHERE id>10")
    db.commit()
    db.close()
    a, b = Store(path), Store(path)
    with a.snapshot():
        a.db.execute("SELECT count(*) FROM chunks").fetchone()
        with b.transaction():
            b.set_meta("other", "1")
        assert a.exact_chunks("abc") == {1}
        assert a.exact_chunks("abc", word=True) == {1}
    assert a.get_meta("fts_reset") == "1"
    assert a.fill_fts() is True
    assert a.exact_state() == "ready" and a.get_meta("fts_reset") is None
    check(a)
    assert b.exact_chunks("abc") == {1}
    a.close()
    b.close()


def test_damage_seen_in_a_snapshot_while_another_writes(tmp_path: Path) -> None:
    """When another connection is still writing as the snapshot ends, the
    damage is not recorded then; the next fill_fts() finds it."""
    path = tmp_path / "idx.sqlite"
    one_chunk(path)
    db = sqlite3.connect(path)
    db.execute(f"DELETE FROM {store.FTS_TABLE}_data WHERE id>10")
    db.commit()
    db.close()
    a, b = Store(path), Store(path)
    with b.transaction():
        with a.snapshot():
            assert a.exact_chunks("abc") == {1}
        b.set_meta("other", "1")
    assert a.get_meta("fts_reset") is None
    assert b.fill_fts() is True
    assert b.get_meta("fts_reset") is None
    check(b)
    assert b.exact_chunks("abc") == {1}
    a.close()
    b.close()


def test_search_in_a_snapshot_while_another_updates(tmp_path: Path) -> None:
    """Without damage, a commit by another connection changes neither the
    snapshot's lookups nor the FTS state, for a writable or read-only store."""
    path = tmp_path / "idx.sqlite"
    one_chunk(path)
    for ro in (False, True):
        a, b = Store(path, readonly=ro), Store(path)
        before = chunk_ids(b)
        with a.snapshot():
            assert a.exact_chunks("abc") == before
            with b.transaction():
                b.put_file(f"/n/b{ro}.txt", "text", 1, 1, "t", None, "ok", None, [chunk("abc x")])
            assert a.exact_chunks("abc", word=True) == before
        assert a.exact_chunks("abc") == chunk_ids(b) != before
        assert b.get_meta("fts_reset") is None
        check(b)
        a.close()
        b.close()
