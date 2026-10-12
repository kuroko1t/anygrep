"""Draw demo/image-search.png: a query image, the image anygrep finds for it, and
the matching result line.

    python scripts/make_demo_figure.py CORPUS INDEX [OUT]

CORPUS is a folder made by scripts/make_demo_corpus.py and INDEX an index of
it (scripts/record_demo.sh makes both). The search really runs, so the path
and score are the ones `anygrep --image query.png -m 1` prints in CORPUS.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import anygrep

WIDTH = 888  # the width of demo/anygrep.gif
PAD = 24
BG = (39, 40, 34)  # the GIF's terminal colours (agg's monokai theme)
FG = (248, 248, 242)
DIM = (160, 160, 150)
GREEN = (166, 226, 46)


def mono(size: int, bold: bool = False) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    names = ["DejaVuSansMono-Bold.ttf", "Menlo.ttc", "consolab.ttf"] if bold else []
    names += ["DejaVuSansMono.ttf", "Menlo.ttc", "consola.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def main() -> None:
    corpus = Path(sys.argv[1]).resolve()
    index = Path(sys.argv[2])
    out = Path(sys.argv[3] if len(sys.argv) > 3 else "demo/image-search.png")
    query = corpus / "query.png"
    with anygrep.Index(index, create=False) as idx:
        hit = idx.search(image=query, k=1)[0]
    found = Path(hit.path).resolve()
    rel = found.relative_to(corpus).as_posix()

    font, bold = mono(20), mono(20, bold=True)
    thumb_w = (WIDTH - 3 * PAD - 40) // 2
    thumbs = []
    for p in (query, found):
        im = Image.open(p).convert("RGB")
        thumbs.append(im.resize((thumb_w, round(im.height * thumb_w / im.width)), Image.LANCZOS))
    thumb_h = max(t.height for t in thumbs)
    line_h = 28
    height = PAD + line_h + thumb_h + PAD + 2 * line_h + PAD
    fig = Image.new("RGB", (WIDTH, height), BG)
    d = ImageDraw.Draw(fig)

    # Captions, the two images and an arrow between them.
    x_right = WIDTH - PAD - thumb_w
    d.text((PAD, PAD), "query.png", font=font, fill=DIM)
    d.text((WIDTH - PAD - d.textlength(rel, font=font), PAD), rel, font=font, fill=DIM)
    y = PAD + line_h
    fig.paste(thumbs[0], (PAD, y))
    fig.paste(thumbs[1], (x_right, y))
    mid_y = y + thumb_h // 2
    x0, x1 = PAD + thumb_w + 8, x_right - 8
    d.line((x0, mid_y, x1 - 6, mid_y), fill=FG, width=3)
    d.polygon([(x1, mid_y), (x1 - 12, mid_y - 8), (x1 - 12, mid_y + 8)], fill=FG)

    # The terminal lines, laid out like the GIF.
    y += thumb_h + PAD
    prompt = "$ "
    d.text((PAD, y), prompt, font=bold, fill=GREEN)
    d.text(
        (PAD + d.textlength(prompt, font=bold), y),
        "anygrep --image query.png -m 1",
        font=font,
        fill=FG,
    )
    y += line_h
    head = f" 1  {hit.score:.3f}  "
    d.text((PAD, y), head, font=font, fill=FG)
    x = PAD + d.textlength(head, font=font)
    d.text((x, y), rel, font=bold, fill=FG)
    x += d.textlength(rel + "  ", font=bold)
    d.text((x, y), "[image]", font=font, fill=DIM)

    out.parent.mkdir(parents=True, exist_ok=True)
    small = fig.quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    small.save(out, optimize=True)
    print(f"Wrote {out} ({rel}, {hit.score:.3f})")


if __name__ == "__main__":
    main()
