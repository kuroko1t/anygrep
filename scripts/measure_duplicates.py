"""Measure the near-duplicate rule used to collapse search results.

Builds the calibration corpus and the held-out corpora, adds deliberate
copies (exact copies, a plain-text export, edited versions, a regenerated PDF,
resized, recompressed and cropped screenshots and photos) and different but
related files (the same chart, app screen or template with other data),
indexes everything with the real model and reports, for every pair of files
of the same type:

- the cosine between their mean chunk vectors, and
- the highest cosine between any two of their chunks (an upper bound of the
  best-chunk cosine a query can produce),

for the deliberate copies and for all other pairs. It then runs the
calibration text queries with collapsing on and lists every group formed.

    python scripts/measure_duplicates.py --work DIR [--threads 2]
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib = _load_script("calibrate_thresholds")
heldout = _load_script("heldout_corpus")
demo = calib.demo

STANDUP = """# Weekly status {date}

## Done
{done}

## Next
{next}

## Blockers
{blockers}
"""

PAYSLIP = """Payslip {m} 2026
Employer: Acme Robotics KK
Employee: A. Example
Base salary: 420,000 JPY
Overtime: {o:,} JPY
Income tax: {t:,} JPY
Net pay: {n:,} JPY
"""


def build(root: Path) -> list[tuple[str, str, str]]:
    """Write the corpus; return the deliberate copies as (original, copy, how)."""
    base = root / "base"
    calib.build_corpus(base)
    for name in heldout.CORPORA:
        heldout.build(root / f"heldout-{name}", name)

    d = root / "copies"
    pairs: list[tuple[str, str, str]] = []

    def text_copy(src: str, dst: str, how: str, edit=lambda t: t) -> None:
        t = (root / src).read_text(encoding="utf-8")
        p = d / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(edit(t), encoding="utf-8")
        pairs.append((src, f"copies/{dst}", how))

    def strip_md(t: str) -> str:
        return "\n".join(ln.lstrip("#").strip() for ln in t.splitlines()) + "\n"

    text_copy("base/notes/sourdough.md", "archive/sourdough.md", "exact copy, same name")
    text_copy("base/notes/sourdough.md", "sourdough (copy).md", "exact copy, other name")
    text_copy("base/notes/kyoto-trip.md", "export/kyoto-trip.txt", "plain-text export", strip_md)
    text_copy(
        "base/notes/meetings/2026-03-budget.md",
        "drafts/2026-03-budget-v2.md",
        "edited: a line changed, a line added",
        lambda t: t.replace("15 percent", "12 percent") + "- Follow-up meeting in April.\n",
    )
    text_copy(
        "base/notes/finance/tax-checklist.md",
        "old/tax-checklist-2025.md",
        "older version: a line removed",
        lambda t: "\n".join(t.splitlines()[:-2]) + "\n",
    )
    text_copy(
        "base/notes/finance/kakutei-shinkoku.md",
        "jp/kakutei-shinkoku-memo.md",
        "Japanese, a sentence added",
        lambda t: t + "医療費の領収書は5年間保管する。\n",
    )
    text_copy(
        "heldout-recipes/recipes/pancakes.md",
        "recipes/pancakes-double.md",
        "edited: quantities changed",
        lambda t: t.replace("200 g", "400 g").replace("2 tsp", "4 tsp"),
    )

    pdf_name, pages = next(iter(demo.PDFS.items()))
    (d / "docs").mkdir(parents=True, exist_ok=True)
    shutil.copy(root / "base" / pdf_name, d / "docs" / Path(pdf_name).name)
    pairs.append((f"base/{pdf_name}", f"copies/docs/{Path(pdf_name).name}", "exact PDF copy"))
    rev = [list(p) for p in pages]
    rev[0] = rev[0] + ["Revised edition, 2026."]
    rev_name = Path(pdf_name).stem + "-rev2.pdf"
    demo.write_pdf(d / "docs" / rev_name, rev)
    pairs.append((f"base/{pdf_name}", f"copies/docs/{rev_name}", "PDF regenerated, line added"))

    def img_copy(src: str, dst: str, how: str, edit, **save) -> None:
        im = Image.open(root / src)
        im.load()
        p = d / dst
        p.parent.mkdir(parents=True, exist_ok=True)
        edit(im.convert("RGB")).save(p, **save)
        pairs.append((src, f"copies/{dst}", how))

    def scale(f: float):
        return lambda im: im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)

    def crop(f: float):
        def c(im):
            dx, dy = round(im.width * f), round(im.height * f)
            return im.crop((dx, dy, im.width - dx, im.height - dy))

        return c

    img_copy(
        "base/screenshots/error-dialog.png", "shots/error-dialog-small.png", "resized 50%",
        scale(0.5),
    )  # fmt: skip
    img_copy(
        "base/screenshots/terminal.png", "shots/terminal.jpg", "PNG -> JPEG q70",
        lambda im: im, quality=70,
    )  # fmt: skip
    img_copy(
        "base/photos/sunset-beach.jpg", "photos/sunset-cropped.jpg", "cropped 4% each side",
        crop(0.04), quality=88,
    )  # fmt: skip
    img_copy(
        "base/photos/whiteboard.jpg", "photos/whiteboard-edit.jpg", "resized 75%, brighter",
        lambda im: ImageEnhance.Brightness(scale(0.75)(im)).enhance(1.1), quality=85,
    )  # fmt: skip
    shutil.copy(root / "base/screenshots/electricity-usage.png", d / "shots" / "chart.png")
    pairs.append(("base/screenshots/electricity-usage.png", "copies/shots/chart.png", "exact copy"))
    img_copy(
        "heldout-mixed/images/weather-app.png", "shots/weather-thumb.png", "resized 33%",
        scale(0.33),
    )  # fmt: skip

    img_copy(
        "base/diagrams/login-flow.png", "shots/login-flow-thumb.jpg", "resized 25%, JPEG q50",
        scale(0.25), quality=50,
    )  # fmt: skip
    img_copy(
        "base/photos/red-bicycle.jpg", "photos/red-bicycle-crop.jpg", "cropped 8% each side",
        crop(0.08), quality=90,
    )  # fmt: skip
    img_copy(
        "heldout-mixed/images/receipt.jpg", "photos/receipt-q40.jpg", "JPEG q40",
        lambda im: im, quality=40,
    )  # fmt: skip

    # Long notes (several chunks): an edited copy, and a note that shares only
    # its first section with the original.
    def joined(*names: str) -> str:
        return "\n".join(demo.NOTES[f"notes/{n}"] for n in names)

    long_text = joined("kyoto-trip.md", "health/half-marathon-plan.md", "reading/habits.md",
                       "home/houseplants.md", "dev/docker-cheatsheet.md")  # fmt: skip
    (root / "base" / "notes" / "journal-2026.md").write_text(long_text, encoding="utf-8")
    text_copy(
        "base/notes/journal-2026.md",
        "journal-2026-edited.md",
        "long note, one section edited",
        lambda t: (
            t.replace("Arashiyama bamboo grove at 7am", "Arashiyama bamboo grove at 8am")
            + "\nAdded later: book the hotel early for November.\n"
        ),
    )
    text_copy(
        "base/notes/reading/habits.md",
        "reading/habits-extended.md",
        "a paragraph appended",
        lambda t: (
            t + "\nA second pass a month later: the two-minute rule worked best for "
            "starting, and tracking streaks on paper kept me going on busy days.\n"
        ),
    )

    # Different but related files: must stay separate.
    rel = root / "related"
    (rel / "images").mkdir(parents=True, exist_ok=True)
    demo.bar_chart(
        "Monthly water usage (m3)",
        [9, 8, 9, 10, 12, 14, 15, 15, 12, 10, 9, 9],
        calib.MONTHS,
        (220, 120, 40),
    ).save(rel / "images" / "water-usage.png")
    demo.bar_chart(
        "Monthly electricity usage (kWh)",
        [430, 400, 360, 310, 250, 230, 240, 260, 290, 340, 390, 440],
        calib.MONTHS,
        (60, 120, 200),
    ).save(rel / "images" / "electricity-usage-2025.png")
    heldout.weather_app(city="Paris", seed=7, accent=(220, 90, 40)).save(
        rel / "images" / "weather-paris.png"
    )
    heldout.chessboard(light=(235, 235, 210), dark=(110, 150, 90), seed=9).save(
        rel / "images" / "chess-green.png"
    )
    heldout.receipt(
        store="CORNER SHOP", items=[("Apples", 3.20), ("Rice 2kg", 4.50), ("Tea", 2.80)], seed=4
    ).save(rel / "images" / "receipt-corner.jpg", quality=90)
    heldout.org_chart(
        names={"CEO": "Tom Weber", "CTO": "Yuki Sato", "CFO": "Lena Novak"}, color=(230, 245, 225)
    ).save(rel / "images" / "org-chart-2.png")
    (rel / "notes").mkdir(parents=True, exist_ok=True)
    weeks = [
        ("2026-05-04", "- Shipped the login page\n- Fixed the CSV export", "- Start on billing",
         "- Waiting for the design review"),
        ("2026-05-11", "- Billing API draft\n- Interviewed two candidates", "- Payment provider "
         "sandbox", "- None"),
    ]  # fmt: skip
    for date, done, nxt, blk in weeks:
        (rel / "notes" / f"status-{date}.md").write_text(
            STANDUP.format(date=date, done=done, next=nxt, blockers=blk), encoding="utf-8"
        )
    (rel / "notes" / "journal-2025.md").write_text(
        joined("kyoto-trip.md", "dev/rust-borrow-checker.md", "recipes/vegetable-curry.md",
               "finance/tax-checklist.md", "sourdough.md"),
        encoding="utf-8",
    )  # fmt: skip
    t = (root / "base/notes/sourdough.md").read_text(encoding="utf-8")
    (rel / "notes" / "sourdough-rye.md").write_text(
        t.replace("Sourdough bread, the version", "Rye sourdough, the version")
        .replace("450 g bread flour, 50 g whole wheat", "300 g rye flour, 200 g bread flour")
        .replace("350 g water", "400 g water")
        .replace("tight boule", "loaf tin")
        .replace("Dutch oven at 250 C", "loaf tin at 240 C"),
        encoding="utf-8",
    )
    # Monthly forms: the same labels, another month and other amounts.
    forms = rel / "forms"
    forms.mkdir(parents=True, exist_ok=True)
    for i, (month, o) in enumerate([("January", 12500), ("February", 40000), ("March", 0)]):
        t = 30000 + o // 10
        (forms / f"payslip-2026-{i + 1:02d}.txt").write_text(
            PAYSLIP.format(m=month, o=o, t=t, n=420000 + o - t), encoding="utf-8"
        )
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json", type=Path, default=ROOT / "docs" / "duplicate-results.json")
    args = ap.parse_args()

    from anygrep.config import model_dir
    from anygrep.embedder import OnnxEmbedder
    from anygrep.indexer import IndexOptions, run_index
    from anygrep.search import DUP_MAX_CHANGED, DUP_MIN_COSINE, search
    from anygrep.signature import changed_share
    from anygrep.store import Store

    root = (args.work / "corpus").resolve()
    if root.exists():
        shutil.rmtree(root)
    copies = build(root)
    embedder = OnnxEmbedder(model_dir(), threads=args.threads)
    store = Store(args.work / "duplicates.sqlite")
    run_index(store, [root], IndexOptions(image_budget=140, quiet=True), lambda v: embedder)

    _, mat, paths, kinds = store.load_vectors()
    rel = [Path(p).relative_to(root).as_posix() for p in paths]
    rows: dict[str, list[int]] = {}
    for i, r in enumerate(rel):
        rows.setdefault(r, []).append(i)
    kind = {r: kinds[ix[0]] for r, ix in rows.items()}
    info = store.file_info(paths)
    by_rel = {Path(p).relative_to(root).as_posix(): v for p, v in info.items()}

    def pair_stats(a: str, b: str) -> dict:
        sha_a, ma, sig_a = by_rel[a]
        sha_b, mb, sig_b = by_rel[b]
        best = float((mat[rows[a]] @ mat[rows[b]].T).max())
        changed = None
        if kind[a] in ("text", "pdf"):
            c = changed_share(sig_a, sig_b)
            changed = None if c is None else round(c, 4)
        return {
            "a": a, "b": b, "type": kind[a], "same_sha": sha_a == sha_b,
            "mean_cos": round(float(ma @ mb), 4), "max_chunk_cos": round(best, 4),
            "changed": changed,
        }  # fmt: skip

    cluster = {r: r for r in rows}

    def find(r: str) -> str:
        while cluster[r] != r:
            r = cluster[r]
        return r

    for a, b, _ in copies:
        cluster[find(b)] = find(a)
    pos = [dict(pair_stats(a, b), how=how) for a, b, how in copies]
    neg = [
        pair_stats(a, b)
        for a, b in itertools.combinations(sorted(rows), 2)
        if kind[a] == kind[b] and find(a) != find(b)
    ]

    def merges(p: dict) -> bool:
        t = DUP_MIN_COSINE[p["type"]]
        words_ok = p["changed"] is None or p["changed"] <= DUP_MAX_CHANGED
        return p["same_sha"] or (min(p["mean_cos"], p["max_chunk_cos"]) >= t and words_ok)

    print(f"{len(rows)} files, {len(pos)} deliberate copies, {len(neg)} other same-type pairs\n")
    print("Deliberate copies (mean cos / max chunk cos):")
    for p in pos:
        v = min(p["mean_cos"], p["max_chunk_cos"])
        r = "" if p["changed"] is None else f"  changed {p['changed']:.3f}"
        print(
            f"  {p['type']:5s} {v:.4f}  {p['how']:38s} {p['b']}{'  [same sha]' * p['same_sha']}{r}"
        )
    report: dict = {
        "rule": dict(DUP_MIN_COSINE),
        "max_changed": DUP_MAX_CHANGED,
        "copies": pos,
        "other_pairs_by_type": {},
    }
    print("\nOther pairs, highest min(mean cos, max chunk cos) per type:")
    for k in ("text", "pdf", "image"):
        sel = sorted(
            (p for p in neg if p["type"] == k),
            key=lambda p: -min(p["mean_cos"], p["max_chunk_cos"]),
        )
        pk = [p for p in pos if p["type"] == k and not p["same_sha"]]
        lo = min((min(p["mean_cos"], p["max_chunk_cos"]) for p in pk), default=None)
        hi = min(sel[0]["mean_cos"], sel[0]["max_chunk_cos"]) if sel else None
        report["other_pairs_by_type"][k] = {
            "pairs": len(sel), "lowest_copy": lo, "highest_other": hi, "top": sel[:8]
        }  # fmt: skip
        print(f"  {k}: {len(sel)} pairs; lowest copy {lo}, highest other {hi}")
        for p in sel[:5]:
            r = "" if p["changed"] is None else f"  changed {p['changed']:.3f}"
            print(f"     {p['mean_cos']:.4f} / {p['max_chunk_cos']:.4f}  {p['a']}  ~  {p['b']}{r}")

    missed = [p["b"] for p in pos if not merges(p)]
    merged = [(p["a"], p["b"]) for p in neg if merges(p)]
    print(f"\nAt {DUP_MIN_COSINE}: copies not merged {missed}, other pairs merged {merged}")
    report["copies_not_merged"], report["other_pairs_merged"] = missed, merged

    # End to end: the calibration queries with collapsing on. A merge is
    # wrong when its files are not copies of the same original.
    groups: dict[tuple, int] = {}
    wrong = 0
    qvecs = [embedder.embed_query(q) for q, _ in calib.TEXT_QUERIES]
    qvecs += [embedder.embed_query(q) for q in calib.NEGATIVE_QUERIES]
    n_text = len(qvecs)
    qvecs += [embedder.embed_image(make()) for _, make, _ in calib.IMAGE_QUERIES]
    timing = {False: 0.0, True: 0.0}
    for j, qv in enumerate(qvecs):
        qk = "text" if j < n_text else "image"
        for c in (False, True):
            t0 = time.perf_counter()
            hits = search(store, qv, n=10, query_kind=qk, hide_weak=False, collapse=c)
            timing[c] += time.perf_counter() - t0
        for h in hits:
            if not h.similar:
                continue
            members = tuple(Path(x.path).relative_to(root).as_posix() for x in [h, *h.similar])
            groups[members] = groups.get(members, 0) + 1
            wrong += sum(find(m) != find(members[0]) for m in members[1:])
    ms = {k: round(1000 * v / len(qvecs), 2) for k, v in timing.items()}
    print(
        f"\nSearch time per query (ms, {len(rows)} files): plain {ms[False]}, collapse {ms[True]}"
    )
    report["search_ms"] = {"plain": ms[False], "collapse": ms[True]}
    print(f"Groups formed over {len(qvecs)} queries (top 10, weak included):")
    for g, c in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {c:3d}x  {' + '.join(g)}")
    report["groups"] = [{"members": list(g), "queries": c} for g, c in groups.items()]
    report["wrong_merges"] = wrong
    print(f"Merges outside the deliberate copies: {wrong}")
    args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.json}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
