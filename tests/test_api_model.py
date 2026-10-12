"""anygrep.Index with the real EmbeddingGemma 2 model. Skipped when the model
files are not present locally (no download is attempted from tests)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import anygrep
from anygrep.model import missing_files

pytestmark = pytest.mark.model

if missing_files(vision=True):
    pytest.skip("model files not available locally", allow_module_level=True)


def test_index_search_with_the_model(tmp_path: Path) -> None:
    root = tmp_path / "notes"
    root.mkdir()
    (root / "bread.md").write_text(
        "# Sourdough\nMix flour, water, salt and starter, proof overnight, bake at 250 C.\n",
        encoding="utf-8",
    )
    (root / "taxes.md").write_text(
        "# Taxes\nThe tax return is due on March 15. Collect the receipts first.\n",
        encoding="utf-8",
    )
    im = Image.new("RGB", (640, 480), (235, 240, 245))
    d = ImageDraw.Draw(im)
    for cx in (200, 440):
        d.ellipse((cx - 90, 250, cx + 90, 430), outline=(20, 20, 20), width=10)
    d.line(((200, 340), (300, 220), (440, 340), (200, 340)), fill=(200, 30, 40), width=12)
    im.save(root / "bike.png")

    with anygrep.Index(tmp_path / "idx.sqlite", threads=2) as idx:
        r = idx.update(root)
        assert (r.new, r.failed) == (3, 0)
        first = idx._embedder(True)

        hits = idx.search("when is the tax return due", k=3)
        assert Path(hits[0].path).name == "taxes.md" and not hits[0].weak
        assert "March 15" in hits[0].snippet
        hits = idx.search("a drawing of a bicycle", type="image")
        assert [Path(h.path).name for h in hits] == ["bike.png"]
        assert all(h.weak for h in idx.search("quarterly earnings call", weak=True))

        # The model was loaded once and kept.
        assert idx._embedder(True) is first
