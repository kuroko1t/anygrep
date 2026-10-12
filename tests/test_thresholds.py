from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anygrep import cli, thresholds
from anygrep.thresholds import min_score, weak_mask

from .conftest import FakeEmbedder


def test_min_score_per_pair_and_budget() -> None:
    assert min_score("text", "text", 70) == min_score("text", "text", 280)
    assert min_score("text", "image", 140) == thresholds.MIN_SCORES["text->image"][140]
    # unknown budgets use the nearest measured one
    assert min_score("image", "image", 100) == thresholds.MIN_SCORES["image->image"][70]
    assert min_score("image", "image", 1000) == thresholds.MIN_SCORES["image->image"][280]
    assert min_score("audio", "text") == 0.0


def test_weak_mask_uses_type_cutoff(monkeypatch) -> None:
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->text", {0: 0.5})
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.3})
    scores = np.array([0.6, 0.4, 0.35, 0.2], np.float32)
    kinds = ["text", "text", "image", "image"]
    assert weak_mask(scores, kinds).tolist() == [False, True, False, True]


def _unit(i: int, dim: int = 768) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    v[i] = 1
    return v


@pytest.fixture
def fixed_reference(monkeypatch):
    """A reference set whose median score is `level` for the query _query(level)."""
    ref = {"text": np.stack([_unit(0)] * 5), "image_140": np.stack([_unit(1)] * 5)}
    monkeypatch.setattr(thresholds, "_reference", lambda: ref)
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->text", {0: 0.6})
    monkeypatch.setitem(thresholds.MIN_MARGINS, "text->text", 0.1)


def _query(level: float) -> np.ndarray:
    q = _unit(0) * level + _unit(2) * np.sqrt(1 - level**2)
    return q.astype(np.float32)


def test_reference_margin_hides_generic_queries(fixed_reference) -> None:
    kinds = ["text"] * 3
    scores = np.array([0.72, 0.66, 0.61], np.float32)
    # A specific query: low reference level, only the minimum score applies.
    assert thresholds.thresholds(_query(0.45), kinds)["text"] == pytest.approx(0.6)
    assert weak_mask(scores, kinds, qvec=_query(0.45)).tolist() == [False, False, False]
    # A query that scores high against everything ("asdf"): level 0.6, so a
    # result needs 0.7.
    assert thresholds.thresholds(_query(0.6), kinds)["text"] == pytest.approx(0.7)
    assert weak_mask(scores, kinds, qvec=_query(0.6)).tolist() == [False, True, True]


def test_single_topic_folder_keeps_its_matches(fixed_reference) -> None:
    # 8 notes about the query's topic and 2 unrelated ones. The old rule
    # (0.04 above the median of the indexed chunks) marked all 8 as weak.
    scores = np.array([0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70, 0.69, 0.50, 0.48], np.float32)
    weak = weak_mask(scores, ["text"] * 10, qvec=_query(0.5))
    assert weak.tolist() == [False] * 8 + [True] * 2


def test_shipped_reference_matches_the_model() -> None:
    thresholds._reference.cache_clear()
    ref = thresholds._reference()
    assert ref is not None, "reference.npz is missing or was built for another model revision"
    assert ref["text"].shape[1] == 768 and len(ref["text"]) >= 32
    for b in (70, 140, 280):
        assert ref[f"image_{b}"].shape[1] == 768
    norms = np.linalg.norm(ref["text"], axis=1)
    assert np.allclose(norms, 1, atol=1e-2)


def test_rule_matches_published_calibration() -> None:
    root = Path(__file__).resolve().parents[1]
    rule = json.loads((root / "docs" / "calibration-results.json").read_text("utf-8"))["rule"]
    published = {k: {int(b): t for b, t in v.items()} for k, v in rule["min_scores"].items()}
    assert published == thresholds.MIN_SCORES
    assert rule["min_margins"] == thresholds.MIN_MARGINS


def test_search_uses_query_type_and_budget(tmp_path: Path, corpus: Path, monkeypatch) -> None:
    from anygrep.indexer import IndexOptions, run_index
    from anygrep.search import search
    from anygrep.store import Store

    emb = FakeEmbedder()
    store = Store(tmp_path / "idx.sqlite")
    run_index(store, [corpus], IndexOptions(quiet=True), lambda vision: emb)
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.0})
    monkeypatch.setitem(thresholds.MIN_SCORES, "image->image", {140: 0.0, 280: 2.0})
    q = emb.embed_query("red picture image")

    def flags(**kw) -> list[bool]:
        hits = search(store, q, n=10, kinds=["image"], **kw)
        assert len(hits) == 2
        return [h.weak for h in hits]

    assert flags(query_kind="text", budget=280) == [False, False]
    assert flags(query_kind="image", budget=140) == [False, False]
    assert flags(query_kind="image", budget=280) == [True, True]


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch, corpus: Path, capsys):
    monkeypatch.setenv("ANYGREP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANYGREP_DB", raising=False)
    monkeypatch.chdir(corpus)
    emb = FakeEmbedder()
    monkeypatch.setattr(cli, "_make_embedder_factory", lambda args: lambda vision: emb)
    for pair in ("text->text", "text->pdf"):
        monkeypatch.setitem(thresholds.MIN_SCORES, pair, {0: 0.3})
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.3})
    assert cli.main(["index", str(corpus), "-q"]) == 0
    capsys.readouterr()


def _jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def test_weak_results_hidden_by_default(cli_env, capsys, monkeypatch) -> None:
    assert cli.main(["wifi password", "--json"]) == 0
    out = _jsonl(capsys.readouterr().out)
    assert [h["path"].endswith("manual.pdf") for h in out] == [True]
    assert out[0]["weak"] is False and out[0]["score"] > 0.3

    assert cli.main(["wifi password", "--weak", "--json"]) == 0
    out = _jsonl(capsys.readouterr().out)
    assert len(out) == 5 and out[0]["path"].endswith("manual.pdf")
    assert [h["weak"] for h in out] == [False, True, True, True, True]

    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    assert cli.main(["wifi password", "--weak"]) == 0
    text = capsys.readouterr().out
    assert text.count("(weak)") == 4 and "No strong matches" not in text


def test_no_strong_match_shows_closest_few(cli_env, capsys, monkeypatch) -> None:
    # Only on a terminal; the exit status still says nothing passed the cutoff.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    assert cli.main(["zebra migration"]) == 1
    text = capsys.readouterr().out
    lines = text.splitlines()
    assert lines[0] == cli.NO_STRONG and "No results" not in text
    assert sum("(weak)" in ln for ln in lines) == cli.WEAK_FALLBACK

    # JSON (and pipes) get no fallback: nothing on stdout, exit 1.
    assert cli.main(["zebra migration", "--json", "-m", "2"]) == 1
    cap = capsys.readouterr()
    assert cap.out == "" and "no strong matches" in cap.err
    assert cli.main(["zebra migration", "--json", "-m", "2", "--weak"]) == 0
    out = _jsonl(capsys.readouterr().out)
    assert len(out) == 2 and all(h["weak"] for h in out)
