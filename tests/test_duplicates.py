from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anygrep import cli
from anygrep import search as search_mod
from anygrep.indexer import IndexOptions, run_index
from anygrep.search import is_near_duplicate, search
from anygrep.store import Store

from .conftest import FakeEmbedder, write_pdf


def _add_copies(corpus: Path) -> None:
    bread = (corpus / "notes" / "bread.md").read_text(encoding="utf-8")
    (corpus / "archive").mkdir()
    shutil.copy(corpus / "notes" / "bread.md", corpus / "archive" / "bread.md")  # same content
    # An edited version: one word added, so a different hash but almost the same vector.
    (corpus / "archive" / "bread-v2.md").write_text(bread + "Enjoy.\n", encoding="utf-8")
    # Related but different: shares a few words with the bread note.
    (corpus / "notes" / "cake.md").write_text(
        "# Cake\n\nA sponge cake needs flour eggs sugar and butter.\nBake it in a warm oven.\n",
        encoding="utf-8",
    )
    # Same picture, resized and saved as JPEG: different hash, same (fake) vector.
    Image.open(corpus / "red.png").resize((100, 75)).save(corpus / "archive" / "red-small.jpg")
    # The same text as a different type is not collapsed.
    write_pdf(corpus / "bread.pdf", [bread.splitlines()])


@pytest.fixture
def store(tmp_path: Path, corpus: Path, fake_embedder: FakeEmbedder):
    _add_copies(corpus)
    st = Store(tmp_path / "idx.sqlite")
    run_index(st, [corpus], IndexOptions(quiet=True), lambda v: fake_embedder)
    yield st
    st.close()


def _rel(p: str) -> str:
    return "/".join(Path(p).parts[-2:])


def _groups(hits) -> set[frozenset[str]]:
    return {frozenset(_rel(x.path) for x in [h, *h.similar]) for h in hits}


def test_rule() -> None:
    assert is_near_duplicate("text", "abc", "abc", 0.1, 0.1)  # same hash
    assert not is_near_duplicate("text", None, None, 0.5, 0.5)
    # Images: only the same bytes, however close the vectors.
    assert is_near_duplicate("image", "a", "a", 0.1, 0.1)
    assert not is_near_duplicate("image", "a", "b", 1.0, 1.0)
    assert not is_near_duplicate("audio", "a", "b", 1.0, 1.0)  # unknown type: hash only


def test_copies_are_collapsed(store: Store, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed_query("sourdough bread starter")
    plain = search(store, q, n=10)
    names = [Path(h.path).name for h in plain]
    assert names.count("bread.md") == 2 and "bread-v2.md" in names
    assert all(not h.similar for h in plain)

    hits = search(store, q, n=10, collapse=True)
    groups = _groups(hits)
    assert {"notes/bread.md", "archive/bread.md", "archive/bread-v2.md"} in groups
    assert {"corpus/bread.pdf"} in groups  # other type
    assert {"notes/cake.md"} in groups  # related, not a copy
    assert {"corpus/red.png"} in groups and {"archive/red-small.jpg"} in groups
    assert len(hits) == len(plain) - 2


def test_only_identical_images_are_collapsed(
    store: Store, fake_embedder: FakeEmbedder, corpus: Path
) -> None:
    """A resized copy (the same vector here) stays a result of its own; a
    copy with the same bytes is grouped."""
    q = fake_embedder.embed_query("red picture image")
    hits = search(store, q, n=10, kinds=["image"], collapse=True)
    assert _groups(hits) == {
        frozenset({"corpus/red.png"}),
        frozenset({"archive/red-small.jpg"}),
        frozenset({"corpus/blue.jpg"}),
    }
    shutil.copy(corpus / "red.png", corpus / "archive" / "red-copy.png")
    run_index(store, [corpus], IndexOptions(quiet=True), lambda v: fake_embedder)
    hits = search(store, q, n=10, kinds=["image"], collapse=True)
    assert frozenset({"corpus/red.png", "archive/red-copy.png"}) in _groups(hits)


def test_collapsing_fills_up_to_n(store: Store, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed_query("sourdough bread starter")
    plain = search(store, q, n=3, kinds=["text"])
    assert sum(Path(h.path).name.startswith("bread") for h in plain) == 3
    hits = search(store, q, n=3, kinds=["text"], collapse=True)
    assert len(hits) == 3 and len({h.path for h in hits}) == 3
    assert len(hits[0].similar) == 2
    # Fewer distinct files than n: everything is returned once.
    hits = search(store, q, n=50, kinds=["text"], collapse=True)
    assert len(hits) + sum(len(h.similar) for h in hits) == len(
        search(store, q, n=50, kinds=["text"])
    )


def test_chunks_mode_is_not_collapsed(store: Store, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed_query("sourdough bread starter")
    hits = search(store, q, n=50, per_file=False, collapse=True)
    assert all(not h.similar for h in hits)


@pytest.fixture
def env(tmp_path: Path, monkeypatch, corpus: Path, capsys):
    monkeypatch.setenv("ANYGREP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANYGREP_DB", raising=False)
    emb = FakeEmbedder()
    monkeypatch.setattr(cli, "_make_embedder_factory", lambda args: lambda vision: emb)
    _add_copies(corpus)
    monkeypatch.chdir(corpus)
    assert cli.main(["index", str(corpus), "-q"]) == 0
    capsys.readouterr()
    return emb


def _tty(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def test_cli_note_and_no_collapse(env, capsys, monkeypatch) -> None:
    _tty(monkeypatch)
    assert cli.main(["sourdough bread starter", "--weak", "-t", "text"]) == 0
    out = capsys.readouterr().out
    assert out.count("bread.md:") == 1
    notes = [ln.strip() for ln in out.splitlines() if "similar:" in ln]
    bread = next(n for n in notes if "bread-v2.md" in n)
    assert bread.startswith("(+2 similar: ") and bread.endswith(")")
    shown = set(bread[len("(+2 similar: ") : -1].split(", "))
    assert len(shown) == 2 and str(Path("archive/bread-v2.md")) in shown

    assert cli.main(["sourdough bread starter", "--weak", "--no-collapse"]) == 0
    out = capsys.readouterr().out
    assert out.count("bread.md:") == 2 and "similar" not in out


def test_piped_output_lists_every_file(env, capsys) -> None:
    for extra in ([], ["-l"]):
        assert cli.main(["sourdough bread starter", "--weak", "-t", "text", *extra]) == 0
        out = capsys.readouterr().out
        assert out.count("bread.md") == 2 and "bread-v2.md" in out and "similar" not in out


def test_cli_json_lists_members(env, capsys) -> None:
    assert cli.main(["sourdough bread starter", "--weak", "--json", "-t", "text"]) == 0
    out = _jsonl(capsys.readouterr().out)
    top = next(h for h in out if h["path"].endswith("bread.md"))
    assert sorted(Path(x["path"]).name for x in top["similar"]) == ["bread-v2.md", "bread.md"]
    assert {"score", "weak", "type", "line_start"} <= set(top["similar"][0])
    assert [h["rank"] for h in out] == list(range(1, len(out) + 1))
    assert sum(len(h["similar"]) for h in out) == 2  # the red pictures are not in --type text

    assert cli.main(["sourdough bread", "--weak", "--json", "--no-collapse"]) == 0
    out = _jsonl(capsys.readouterr().out)
    assert all(h["similar"] == [] for h in out)


def test_fallback_follows_n_up_to_a_cap(env, capsys, monkeypatch) -> None:
    # The weak fallback is only shown on a terminal, and still means "no match".
    _tty(monkeypatch)

    def weak_lines() -> int:
        return capsys.readouterr().out.count("(weak)")

    assert cli.main(["zebra migration"]) == 1
    assert weak_lines() == cli.WEAK_FALLBACK
    assert cli.main(["zebra migration", "-m", "5"]) == 1
    assert weak_lines() == 5
    monkeypatch.setattr(cli, "WEAK_FALLBACK_MAX", 4)
    assert cli.main(["zebra migration", "-m", "50"]) == 1
    assert weak_lines() == 4


SLIP = (
    "Payslip {m} 2026\nEmployer: Acme Robotics KK\nEmployee: A. Example\n"
    "Base salary: 420,000 JPY\nOvertime: {o:,} JPY\nIncome tax: {t:,} JPY\nNet pay: {n:,} JPY\n"
)


def _slip(m: str, o: int) -> str:
    t = 30000 + o // 10
    return SLIP.format(m=m, o=o, t=t, n=420000 + o - t)


def test_changed_word_pairs() -> None:
    from anygrep.signature import changed_share, signature

    def changed(a: str, b: str) -> float | None:
        return changed_share(signature([a]), signature([b]))

    note = "Feed the starter daily.\nBake at 250 C for 20 minutes.\nCool for an hour.\n"
    assert changed(note, note) == 0.0
    # Added or removed lines are not changes.
    assert changed(note, note + "Enjoy the bread.\n") == 0.0
    assert changed(note, "\n".join(note.splitlines()[:2])) == 0.0
    # Another month of the same form replaces the month and the amounts.
    a, b = _slip("January", 12500), _slip("February", 40000)
    assert changed(a, b) > search_mod.DUP_MAX_CHANGED
    # Japanese counts per character: one of seven character pairs... twice.
    assert changed("領収書を保管する", "領収書を保存する") == pytest.approx(2 / 7)
    assert changed("", note) is None and signature(["..."]) is None


def test_signature_of_a_long_text_is_bounded_and_estimates_well() -> None:
    import random

    from anygrep.signature import SIG_SIZE, changed_share, signature

    rng = random.Random(0)
    words = [f"w{i}" for i in range(3000)]
    base = [rng.choice(words) for _ in range(8000)]
    sig = signature([" ".join(base)])
    assert sig is not None and len(sig) == 4 * SIG_SIZE
    edited = [w if rng.random() > 0.02 else rng.choice(words) for w in base]
    templ = [w if rng.random() > 0.15 else rng.choice(words) for w in base]
    assert changed_share(sig, signature([" ".join(edited)])) < search_mod.DUP_MAX_CHANGED
    assert changed_share(sig, signature([" ".join(templ)])) > search_mod.DUP_MAX_CHANGED
    # Fed in pieces (pages) or at once: the same signature.
    text = " ".join(base)
    assert signature([text[:20000], text[20000:]]) == signature([text])


def test_rule_checks_words_for_text() -> None:
    from anygrep.signature import signature

    a, b = signature([_slip("January", 12500)]), signature([_slip("February", 40000)])
    a2 = signature([_slip("January", 12500) + "Paid.\n"])
    assert not is_near_duplicate("text", "a", "b", 1.0, 1.0, lambda: (a, b))
    assert not is_near_duplicate("pdf", "a", "b", 1.0, 1.0, lambda: (a, b))
    assert is_near_duplicate("text", "a", "b", 1.0, 1.0, lambda: (a, a2))
    assert is_near_duplicate("text", "a", "b", 1.0, 1.0, lambda: (None, b))  # no words stored
    assert is_near_duplicate("text", "x", "x", 0.1, 0.1, lambda: (a, b))  # same hash

    def never():
        raise AssertionError("signatures read although the vectors differ")

    assert not is_near_duplicate("text", "a", "b", 0.5, 0.5, never)
    assert not is_near_duplicate("image", "a", "b", 1.0, 1.0, never)


def test_monthly_forms_stay_separate(
    tmp_path: Path, fake_embedder: FakeEmbedder, monkeypatch
) -> None:
    # The fake vectors of two months differ more than the real ones (0.97), so
    # the cosine is lowered: here only the word check keeps them apart.
    monkeypatch.setitem(search_mod.DUP_MIN_COSINE, "text", 0.5)
    corpus = tmp_path / "c"
    corpus.mkdir()
    months = ["January", "February", "March", "April", "May", "June"]
    for i, m in enumerate(months):
        (corpus / f"payslip-{i + 1:02d}.txt").write_text(_slip(m, 5000 * i), encoding="utf-8")
    (corpus / "payslip-01-copy.txt").write_text(_slip("January", 0) + "\n", encoding="utf-8")
    st = Store(tmp_path / "idx.sqlite")
    run_index(st, [corpus], IndexOptions(quiet=True), lambda v: fake_embedder)
    q = fake_embedder.embed_query("payslip net pay")
    hits = search(st, q, n=10, collapse=True)
    st.close()
    assert len(hits) == 6
    by_name = {Path(h.path).name: [Path(x.path).name for x in h.similar] for h in hits}
    group = by_name.get("payslip-01.txt", by_name.get("payslip-01-copy.txt"))
    assert sorted(group + [k for k, v in by_name.items() if v == group]) == [
        "payslip-01-copy.txt",
        "payslip-01.txt",
    ]


def test_many_copies_do_not_use_up_the_scan(tmp_path: Path, monkeypatch) -> None:
    # Hundreds of identical files (vendored LICENSE files) rank above the
    # distinct ones: they join one group and do not count towards the limit.
    from anygrep.store import ChunkData

    rng = np.random.default_rng(0)

    def unit(v):
        return (v / np.linalg.norm(v)).astype(np.float32)

    q = unit(rng.standard_normal(768))
    lic = unit(q + 0.3 * rng.standard_normal(768))
    st = Store(tmp_path / "idx.sqlite")

    def put(path: str, sha: str, vec) -> None:
        chunk = ChunkData(ord=0, page=None, line_start=1, line_end=1, text="x", vec=vec)
        st.put_file(path, "text", 1, 1, sha, None, "ok", None, [chunk])

    monkeypatch.setattr(search_mod, "DUP_SCAN_FACTOR", 2)
    monkeypatch.setattr(search_mod, "DUP_SCAN_BASE", 10)  # limit 20 for n=5
    with st.transaction():
        for i in range(60):
            put(f"/c/pkg{i}/LICENSE", "same", lic)
        for i in range(20):
            put(f"/c/doc{i}.md", f"h{i}", unit(q * 0.5 + rng.standard_normal(768)))
    hits = search(st, q, n=5, collapse=True)
    assert len(hits) == 5
    assert Path(hits[0].path).name == "LICENSE" and len(hits[0].similar) == 59
    # A folder made only of copies still stops after limit * DUP_ABSORB_FACTOR files.
    monkeypatch.setattr(search_mod, "DUP_ABSORB_FACTOR", 1)
    hits = search(st, q, n=5, collapse=True)
    assert len(hits) == 1 and len(hits[0].similar) < 59
    st.close()


def _templates(corpus: Path, n: int = 20, words: int = 8000) -> None:
    import random

    rng = random.Random(0)
    vocab = [f"w{i}" for i in range(3000)]
    base = [rng.choice(vocab) for _ in range(words)]
    for f in range(n):
        text = " ".join(w if rng.random() > 0.15 else rng.choice(vocab) for w in base)
        (corpus / f"report-{f:02d}.txt").write_text(text, encoding="utf-8")


def test_collapsing_template_documents_is_fast(tmp_path: Path, monkeypatch) -> None:
    # 20 long reports of one template with 15 percent of the words replaced:
    # their vectors match, so every pair needs the word check. Comparing the
    # whole word lists took about 8 s per search here (48 s on prose).
    import time

    monkeypatch.setitem(search_mod.DUP_MIN_COSINE, "text", 0.5)
    corpus = tmp_path / "c"
    corpus.mkdir()
    _templates(corpus)
    emb = FakeEmbedder()
    st = Store(tmp_path / "idx.sqlite")
    run_index(st, [corpus], IndexOptions(quiet=True), lambda v: emb)

    def no_texts(paths):
        raise AssertionError("file texts read at search time")

    monkeypatch.setattr(st, "file_texts", no_texts)
    q = emb.embed_query("w1 w2 w3")
    t = time.perf_counter()
    hits = search(st, q, n=10, collapse=True)
    assert time.perf_counter() - t < 2.0
    assert len(hits) == 10 and all(not h.similar for h in hits)
    st.close()


def test_index_without_signatures_falls_back_to_the_text(
    tmp_path: Path, fake_embedder: FakeEmbedder, monkeypatch
) -> None:
    # Files indexed by anygrep 0.1.0 have no signature stored.
    monkeypatch.setitem(search_mod.DUP_MIN_COSINE, "text", 0.5)
    corpus = tmp_path / "c"
    corpus.mkdir()
    for i, m in enumerate(["January", "February", "March"]):
        (corpus / f"payslip-{i + 1:02d}.txt").write_text(_slip(m, 5000 * i), encoding="utf-8")
    (corpus / "payslip-01-copy.txt").write_text(_slip("January", 0) + "\n", encoding="utf-8")
    st = Store(tmp_path / "idx.sqlite")
    run_index(st, [corpus], IndexOptions(quiet=True), lambda v: fake_embedder)
    q = fake_embedder.embed_query("payslip net pay")
    with_sig = _groups(search(st, q, n=10, collapse=True))
    st.db.execute("UPDATE files SET sig=NULL")
    assert _groups(search(st, q, n=10, collapse=True)) == with_sig
    assert len(with_sig) == 3
    st.close()


@pytest.mark.parametrize("collapse", [False, True])
def test_search_reads_one_snapshot(tmp_path: Path, corpus: Path, fake_embedder, collapse) -> None:
    # Another process re-indexes a file between reading the vectors and the
    # chunk texts: its old chunk ids are gone from the latest state.
    st = Store(tmp_path / "idx.sqlite")
    run_index(st, [corpus], IndexOptions(quiet=True), lambda v: fake_embedder)
    other = Store(tmp_path / "idx.sqlite")
    q = fake_embedder.embed_query("sourdough bread starter")
    name = "load_matrix" if hasattr(Store, "load_matrix") else "load_vectors"
    real = getattr(st, name)

    def load_then_reindex(*a, **kw):
        out = real(*a, **kw)
        for p in [r[0] for r in other.db.execute("SELECT path FROM files WHERE kind='text'")]:
            row = other.get_file(p)
            assert row is not None
            with other.transaction():
                other.put_file(p, "text", row.size, row.mtime_ns, row.sha256, None, "ok", None, [])
        return out

    setattr(st, name, load_then_reindex)
    hits = search(st, q, n=10, collapse=collapse)
    assert any(Path(h.path).name == "bread.md" for h in hits)
    other.close()
    st.close()


def _vector_loads(st: Store, fn) -> int:
    n = [0]

    def trace(sql: str) -> None:
        n[0] += "c.vec" in sql and "JOIN" in sql and "IN (" not in sql

    st.db.set_trace_callback(trace)
    try:
        fn()
    finally:
        st.db.set_trace_callback(None)
    return n[0]


def test_vectors_are_loaded_once_until_the_index_changes(
    tmp_path: Path, corpus: Path, fake_embedder
) -> None:
    st = Store(tmp_path / "idx.sqlite")
    run_index(st, [corpus], IndexOptions(quiet=True), lambda v: fake_embedder)
    q = fake_embedder.embed_query("bread")

    def twice():
        search(st, q, n=5)
        search(st, q, n=3, collapse=True)

    assert _vector_loads(st, twice) == 1
    assert _vector_loads(st, twice) == 0
    other = Store(tmp_path / "idx.sqlite")  # a change from another process
    other.delete_files([other.get_file(str(corpus / "notes" / "bikes.txt")).id])
    other.close()
    hits = []
    assert _vector_loads(st, lambda: hits.extend(search(st, q, n=10))) == 1
    assert all(Path(h.path).name != "bikes.txt" for h in hits)
    st.close()


def test_path_filter_is_evaluated_once_per_file(
    store: Store, fake_embedder: FakeEmbedder, monkeypatch
) -> None:
    calls: list[str] = []
    real = search_mod.path_matcher

    def counting(pattern):
        m = real(pattern)

        def f(p):
            calls.append(p)
            return m(p)

        return f

    monkeypatch.setattr(search_mod, "path_matcher", counting)
    q = fake_embedder.embed_query("wifi password")
    hits = search(store, q, n=10, scope=search_mod.scope_matcher([], ["*.pdf"]))
    assert hits and all(h.path.endswith(".pdf") for h in hits)
    assert len(calls) == len(set(calls))  # manual.pdf has two chunks


def test_zero_results_requested(store: Store, fake_embedder: FakeEmbedder) -> None:
    q = fake_embedder.embed_query("bread")
    for kw in ({}, {"collapse": True}, {"per_file": False}):
        assert search(store, q, n=0, **kw) == []
        assert search(store, q, n=-3, **kw) == []
