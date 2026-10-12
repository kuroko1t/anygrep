from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anygrep import cli, thresholds
from anygrep.search import search
from anygrep.store import ChunkData, Store

from .conftest import FakeEmbedder


def unit(i: int) -> np.ndarray:
    v = np.zeros(768, np.float32)
    v[i] = 1
    return v


def scored(score: float, axis: int = 0) -> np.ndarray:
    return (score * unit(axis) + np.sqrt(1 - score**2) * unit(3)).astype(np.float32)


@pytest.fixture
def store(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(thresholds, "_reference", lambda: None)
    st = Store(tmp_path / "index.sqlite")
    yield st
    st.close()


def add(store: Store, path: str, budget: int | None, vec: np.ndarray) -> None:
    with store.transaction():
        store.put_file(
            path,
            "text" if budget is None else "image",
            1,
            1,
            None,
            budget,
            "ok",
            None,
            [ChunkData(0, vec)],
        )


def test_text_query_uses_each_result_image_cutoff(store):
    add(store, "/old.png", 70, scored(0.625))
    add(store, "/new.png", 280, scored(0.625))
    hits = search(store, unit(0), budget=280)
    assert {h.path: h.weak for h in hits} == {"/old.png": True, "/new.png": False}
    assert [h.path for h in search(store, unit(0), budget=280, hide_weak=True)] == ["/new.png"]


def test_text_query_uses_each_result_reference_set(store, monkeypatch):
    add(store, "/old.png", 70, scored(0.70))
    add(store, "/new.png", 280, scored(0.70))
    monkeypatch.setattr(
        thresholds,
        "_reference",
        lambda: {"text": unit(2)[None], "image_70": scored(0.65)[None], "image_280": unit(2)[None]},
    )
    hits = search(store, unit(0), budget=280)
    assert {h.path: h.weak for h in hits} == {"/old.png": True, "/new.png": False}


def test_image_query_matches_each_result_budget_and_keeps_text_query_budget(store):
    add(store, "/old.png", 70, scored(0.78, 0))
    add(store, "/new.png", 280, scored(0.72, 1))
    add(store, "/note.md", None, scored(0.60, 1))
    hits = search(
        store,
        unit(1),
        query_kind="image",
        budget=280,
        image_queries={70: unit(0)},
        hide_weak=True,
    )
    assert [h.path for h in hits] == ["/old.png", "/new.png", "/note.md"]
    assert [h.score for h in hits] == pytest.approx([0.78, 0.72, 0.60], abs=5e-4)
    assert all(not h.weak for h in hits)


def test_image_query_extra_budget_uses_matching_reference_vector(store, monkeypatch):
    add(store, "/old.png", 70, scored(0.80, 0))
    monkeypatch.setattr(
        thresholds,
        "_reference",
        lambda: {
            "text": unit(2)[None],
            "image_70": scored(0.75, 0)[None],
            "image_280": unit(2)[None],
        },
    )
    hits = search(store, unit(1), query_kind="image", budget=280, image_queries={70: unit(0)})
    assert len(hits) == 1 and hits[0].weak  # matching baseline + .09 is .84


def test_uncalibrated_cross_budget_image_query_is_weak(store):
    add(store, "/old.png", 70, unit(0))
    hits = search(store, unit(0), query_kind="image", budget=280)
    assert len(hits) == 1 and hits[0].weak
    assert search(store, unit(0), query_kind="image", budget=280, hide_weak=True) == []


def test_budget_metadata_is_available_readonly_and_refreshes(store):
    add(store, "/old.png", 70, unit(0))
    assert store.load_matrix().budgets == [70]
    with store.transaction():
        store.db.execute("UPDATE files SET budget=280 WHERE path='/old.png'")
    assert store.load_matrix().budgets == [280]
    readonly = Store(store.path, readonly=True)
    try:
        assert readonly.load_matrix().budgets == [280]
    finally:
        readonly.close()


@pytest.fixture
def mixed_cli(env, tmp_path, monkeypatch, capsys):
    class BudgetEmbedder(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.query_budgets = []
            self.fail_budget = None

        def embed_image(self, image):
            self.query_budgets.append(self.image_budget)
            if self.image_budget == self.fail_budget:
                raise ValueError("test budget failure")
            return super().embed_image(image)

    emb = BudgetEmbedder()
    monkeypatch.setattr(cli, "_make_embedder_factory", lambda args: lambda vision: emb)
    root = tmp_path / "images"
    root.mkdir()
    old, new, copy = [root / name for name in ("old.png", "new.png", "copy.png")]
    query = tmp_path / "query.png"
    for path in (old, new, copy, query):
        Image.new("RGB", (64, 64), (255, 0, 0)).save(path)
    assert cli.main(["index", str(old), "--image-tokens", "70", "-q"]) == 0
    assert cli.main(["index", str(new), str(copy), "--image-tokens", "280", "-q"]) == 0
    capsys.readouterr()
    emb.query_budgets.clear()
    return emb, root, old, new, query


@pytest.mark.parametrize(
    "scope_name, query_name, expected",
    [
        ("root", "query", [280, 70]),
        ("new", "query", [280]),
        ("root", "old", [280]),
    ],
)
def test_cli_embeds_only_scoped_nonquery_image_budgets(
    mixed_cli,
    capsys,
    scope_name,
    query_name,
    expected,
):
    emb, root, old, new, query = mixed_cli
    paths = {"root": root, "old": old, "new": new, "query": query}
    assert (
        cli.main(
            [
                "--image",
                str(paths[query_name]),
                str(paths[scope_name]),
                "--no-index",
                "--json",
                "--weak",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert emb.query_budgets == expected
    assert emb.image_budget == 280


def test_cli_restores_query_budget_when_additional_embedding_fails(mixed_cli, capsys):
    emb, root, _old, _new, query = mixed_cli
    emb.fail_budget = 70
    assert cli.main(["--image", str(query), str(root), "--no-index", "--json"]) == 2
    assert "test budget failure" in capsys.readouterr().err
    assert emb.image_budget == 280
