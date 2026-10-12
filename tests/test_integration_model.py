"""Runs the real EmbeddingGemma 2 model. Skipped when the model files are not
present locally (no download is attempted from tests)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from anygrep.config import model_dir
from anygrep.indexer import IndexOptions, run_index
from anygrep.model import missing_files
from anygrep.search import search
from anygrep.store import Store

pytestmark = pytest.mark.model

if missing_files(vision=True):
    pytest.skip("model files not available locally", allow_module_level=True)


@pytest.fixture(scope="module")
def embedder():
    from anygrep.embedder import OnnxEmbedder

    threads = int(os.environ.get("ANYGREP_THREADS", "2"))
    return OnnxEmbedder(model_dir(), threads=threads)


def test_embeddings_are_normalised(embedder) -> None:
    q = embedder.embed_query("hello world")
    assert q.shape == (768,)
    assert abs(float(np.linalg.norm(q)) - 1) < 1e-4


def test_mixed_corpus_search(tmp_path: Path, embedder) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "bread.md").write_text(
        "# Sourdough\nMix flour, water, salt and starter, proof overnight, bake at 250 C.\n"
    )
    (root / "taxes.md").write_text(
        "# Taxes\nCollect receipts for medical expenses before filing.\n"
    )
    im = Image.new("RGB", (640, 480), (235, 240, 245))
    d = ImageDraw.Draw(im)
    for cx in (200, 440):
        d.ellipse((cx - 90, 250, cx + 90, 430), outline=(20, 20, 20), width=10)
    d.line(((200, 340), (300, 220), (440, 340), (200, 340)), fill=(200, 30, 40), width=12)
    im.save(root / "bike.png")
    store = Store(tmp_path / "idx.sqlite")
    rep = run_index(store, [root], IndexOptions(quiet=True), lambda vision: embedder)
    assert rep.new == 3 and rep.errors == 0
    top = search(store, embedder.embed_query("how to bake bread"), n=1)[0]
    assert top.path.endswith("bread.md")
    top = search(store, embedder.embed_query("a drawing of a bicycle"), n=1)[0]
    assert top.path.endswith("bike.png")


def test_special_token_text_is_plain_text(embedder) -> None:
    from anygrep.config import EOS_ID, IMAGE_ID

    ids = embedder._ids("The processor inserts <|image|> placeholders before <eos>.")
    assert IMAGE_ID not in ids and ids.count(EOS_ID) == 1
    q = embedder.embed_query("what is <|image|> or <|audio|>")
    assert q.shape == (768,)


def test_golden_scores(embedder) -> None:
    """Catches changes to prompts, pooling, tokenizer or runtime that would
    shift every score (and with them the calibrated cutoffs)."""
    q = embedder.embed_query("how long should sourdough proof")
    d = embedder.embed_documents(
        [
            "Let the dough proof overnight in the fridge, then bake at 250 C.",
            "The quarterly tax return is due in April.",
        ],
        ["sourdough", "taxes"],
    )
    assert np.allclose(d @ q, [0.7552, 0.5371], atol=0.01)


@pytest.fixture(scope="module")
def demo_index(tmp_path_factory, embedder):
    root = tmp_path_factory.mktemp("demo") / "corpus"
    _build_demo(root)
    store = Store(root.parent / "idx.sqlite")
    rep = run_index(store, [root], IndexOptions(quiet=True), lambda vision: embedder)
    assert rep.errors == 0 and rep.new > 20
    yield root, store
    store.close()


def _build_demo(root: Path) -> None:
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "make_demo_corpus.py"
    subprocess.run([sys.executable, str(script), str(root)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("reset the wifi password", "docs/home-network-manual.pdf"),
        ("how long should sourdough proof", "notes/sourdough.md"),
        ("temples to visit in Kyoto", "notes/kyoto-trip.md"),
        ("確定申告の期限", "notes/finance/kakutei-shinkoku.md"),
        ("screenshot of a disk full error", "screenshots/error-dialog.png"),
    ],
)
def test_demo_queries_pass_the_cutoff(demo_index, embedder, query: str, expected: str) -> None:
    root, store = demo_index
    hits = search(store, embedder.embed_query(query), n=3, query_kind="text")
    assert Path(hits[0].path).relative_to(root).as_posix() == expected
    assert not hits[0].weak


@pytest.mark.parametrize("query", ["asdf", "hello", "...", "medieval castle siege weapons"])
def test_unrelated_queries_only_get_weak_results(demo_index, embedder, query: str) -> None:
    _, store = demo_index
    hits = search(store, embedder.embed_query(query), n=10, query_kind="text")
    assert hits and all(h.weak for h in hits)


def test_image_query_finds_the_similar_picture(demo_index, embedder) -> None:
    from .conftest import demo

    root, store = demo_index
    embedder.image_budget = 140
    query = demo.bicycle().transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    hits = search(store, embedder.embed_image(query), n=3, query_kind="image")
    assert Path(hits[0].path).relative_to(root).as_posix() == "photos/red-bicycle.jpg"
    assert not hits[0].weak
