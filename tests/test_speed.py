"""Speed changes that must not change results: identical images reuse a
vector, the scope is read from the index only for its folders, and the
ranking is computed lazily."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anygrep import indexer
from anygrep.indexer import IndexOptions, run_index
from anygrep.search import _Ranked, load_prefixes, scope_matcher, search
from anygrep.store import Store

from .conftest import FakeEmbedder


def index(store: Store, roots, emb, **kw):
    return run_index(store, list(roots), IndexOptions(quiet=True, **kw), lambda vision: emb)


class PoisonEmbedder(FakeEmbedder):
    """FakeEmbedder that fails on images of some colours."""

    def __init__(self) -> None:
        super().__init__()
        self.poison: set[tuple[int, int, int]] = set()

    def embed_image(self, image):
        if image.getpixel((0, 0)) in self.poison:
            raise RuntimeError("bad image")
        return super().embed_image(image)


def _img(path: Path, colour, size=(64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


def _vec(store: Store, path: Path) -> bytes:
    fid = store.get_file(str(path)).id
    return store.db.execute("SELECT vec FROM chunks WHERE file_id=?", (fid,)).fetchone()[0]


def test_identical_images_are_embedded_once(tmp_path: Path) -> None:
    d = tmp_path / "pics"
    a = _img(d / "photo.png", (200, 10, 10))
    b = d / "photo (1).png"
    shutil.copyfile(a, b)
    _img(d / "other.png", (10, 200, 10))
    emb = FakeEmbedder()
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [d], emb)
    assert (rep.new, rep.reused, rep.errors) == (3, 1, 0)
    # The copy is not counted as embedded (it would lower the s/file shown).
    assert rep.by_kind_files == {"image": 2}
    assert emb.calls["image"] == 2
    assert _vec(store, a) == _vec(store, b)

    # A later copy, under another name and folder, is taken from the index.
    c = tmp_path / "pics" / "sub" / "renamed.png"
    c.parent.mkdir()
    shutil.copyfile(a, c)
    rep = index(store, [d], emb)
    assert (rep.new, rep.reused) == (1, 1)
    assert emb.calls["image"] == 2
    assert _vec(store, c) == _vec(store, a)


def test_identical_image_at_another_budget_is_embedded(tmp_path: Path, fake_embedder) -> None:
    a = _img(tmp_path / "one" / "a.png", (200, 10, 10))
    store = Store(tmp_path / "idx.sqlite")
    index(store, [a.parent], fake_embedder, image_budget=140)
    b = tmp_path / "two" / "b.png"
    b.parent.mkdir()
    shutil.copyfile(a, b)
    rep = index(store, [b.parent], fake_embedder, image_budget=70)
    assert (rep.new, rep.reused) == (1, 0)
    assert fake_embedder.calls["image"] == 2


def test_reused_images_do_not_calibrate(tmp_path: Path, monkeypatch) -> None:
    """The stored speed factor counts only the images embedded: copies take
    no model time, and counting them would make later estimates too low."""
    monkeypatch.setattr(indexer, "CALIB_MIN_SECONDS", 0.0)
    seen = []
    monkeypatch.setattr(
        indexer, "update_throughput", lambda store, est, secs, key: seen.append(est)
    )
    d = tmp_path / "pics"
    a = _img(d / "a.png", (200, 10, 10))
    for i in range(3):
        shutil.copyfile(a, d / f"copy{i}.png")
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [d], FakeEmbedder())
    assert rep.reused == 3
    assert seen == [pytest.approx(indexer.estimate_cost(_entry(a), 140))]
    assert rep.embedded_cost == pytest.approx(seen[0])


def _entry(path: Path):
    from anygrep.walker import FileEntry

    st = path.stat()
    return FileEntry(path, "image", st.st_size, st.st_mtime_ns)


def test_failed_image_and_its_copy_are_retried(tmp_path: Path) -> None:
    d = tmp_path / "pics"
    for i in range(3):
        _img(d / f"{i}.png", (10 + i, 100, 50))
    bad = _img(d / "bad.png", (1, 2, 3))
    shutil.copyfile(bad, d / "bad copy.png")
    emb = PoisonEmbedder()
    emb.poison = {(1, 2, 3)}
    store = Store(tmp_path / "idx.sqlite")
    rep = index(store, [d], emb)
    assert (rep.new, rep.errors) == (3, 2)
    assert store.get_file(str(bad)).note.startswith("EmbedError: ")
    # Retried on the next run (the model may have failed for another reason).
    emb.poison = set()
    rep = index(store, [d], emb)
    assert (rep.updated, rep.reused, rep.errors) == (2, 1, 0)
    assert _vec(store, bad) == _vec(store, d / "bad copy.png")


def _full_order(scores: np.ndarray) -> list[int]:
    return np.argsort(-scores, kind="stable").tolist()


@pytest.mark.parametrize("seed", range(5))
def test_lazy_ranking_matches_full_sort(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 5000
    # Many exact ties (identical files) and rows filtered out (-inf).
    scores = rng.integers(0, 300, n).astype(np.float32) / 300
    scores[rng.random(n) < 0.3] = -np.inf
    r = _Ranked(scores, 10)
    assert len(r.order) < n  # not sorted in full up front
    assert [r[i] for i in range(n)] == _full_order(scores)


def test_lazy_ranking_with_nan_falls_back_to_full_sort() -> None:
    scores = np.array([0.5, np.nan, 0.7, -np.inf, 0.7], np.float32)
    r = _Ranked(scores, 1)
    assert [r[i] for i in range(len(r))] == _full_order(scores)


def _text_corpus(root: Path) -> list[Path]:
    words = ["apple pie recipe", "bicycle repair", "tax return", "garden tomatoes"]
    out = []
    for d in ["a", "ab", "a/b", "c"]:
        for i, w in enumerate(words):
            p = root / d / f"{w.split()[0]}{i}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{w} notes {d} {i}\n" * 3)
            out.append(p)
    return out


def test_scope_prefilter_gives_the_same_results(tmp_path: Path, fake_embedder) -> None:
    root = tmp_path / "data"
    _text_corpus(root)
    store = Store(tmp_path / "idx.sqlite")
    index(store, [root], fake_embedder)
    q = fake_embedder.embed_query("apple pie")
    for prefixes, globs in [
        ([str(root / "a")], []),
        ([str(root / "a"), str(root / "c")], []),
        ([str(root / "a")], ["!*/b/*"]),
        ([str(root / "a" / "apple0.txt")], []),
        ([str(root)], ["*.txt"]),
    ]:
        scope = scope_matcher(prefixes, globs)
        for kw in ({}, {"collapse": True}, {"per_file": False}, {"hide_weak": True}):
            store._vec_cache = None
            want = search(store, q, n=50, scope=scope, **kw)
            store._vec_cache = None
            got = search(store, q, n=50, scope=scope, prefixes=prefixes, **kw)
            assert [(h.path, h.chunk_id, h.weak) for h in got] == [
                (h.path, h.chunk_id, h.weak) for h in want
            ]
            # A smaller matrix may round the last bit differently (macOS BLAS).
            assert [h.score for h in got] == pytest.approx([h.score for h in want], abs=1e-6)
    # "a" does not take in "ab".
    v = store.load_matrix(None, load_prefixes([str(root / "a")]))
    assert v.paths and all("/ab/" not in p and "/c/" not in p for p in v.paths)


def test_prefiltered_rows_keep_their_order(tmp_path: Path, fake_embedder) -> None:
    root = tmp_path / "data"
    _text_corpus(root)
    store = Store(tmp_path / "idx.sqlite")
    index(store, [root], fake_embedder)
    full = store.load_matrix()
    store._vec_cache = None
    part = store.load_matrix(None, [str(root / "a")])
    ids = full.ids.tolist()
    pos = [ids.index(i) for i in part.ids.tolist()]
    assert pos == sorted(pos) and len(pos) < len(ids)
    assert np.array_equal(full.mat[pos], part.mat)
