"""Measure score distributions per query/result type pair and pick the cutoffs.

Builds a calibration corpus (the demo corpus plus extra synthetic notes, a PDF
and images, all generated here), indexes it with the real model, runs labelled
queries (relevant ones and clearly unrelated ones: off-topic questions,
nonsense strings and very short queries) and reports, per type pair and image
token budget, the score distributions and the precision/recall of the rule in
``anygrep.thresholds`` (a minimum score per pair, plus a margin above the
query's reference level).

The same rule is then checked on held-out corpora (scripts/heldout_corpus.py)
that played no part in choosing the values, so the published numbers are not
only in-sample.

    python scripts/calibrate_thresholds.py [--budgets 140,70,280] [--threads 2]
        [--work DIR] [--json docs/calibration-results.json] [--no-heldout]

The work directory (default: a new temporary directory) must not be inside a
git-ignored folder, or the walker skips the corpus.

A (query, file) pair counts as relevant when the file is in the query's
relevant set; the file score is its best chunk, as in the default search
output. Recall is measured over relevant pairs, precision over all pairs
that pass the cutoff (from every query, including the unrelated ones).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anygrep.thresholds import (  # noqa: E402
    MIN_MARGINS,
    MIN_SCORES,
    min_margin,
    min_score,
    reference_level,
    thresholds,
)

# The rule used before the reference level: a margin above the median score
# of the indexed chunks of the same type. Reported for comparison only.
LEGACY_MARGIN = 0.04
LEGACY_MIN_ROWS = 8


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load_script("make_demo_corpus")
font = demo.font

# ---------------------------------------------------------------- extra corpus items
EXTRA_NOTES = {
    "notes/dev/git-tips.md": """# Git tips

- Undo the last commit but keep the changes: `git reset --soft HEAD~1`
- Find which commit broke the build: `git bisect start`, then mark good/bad.
- Clean up merged branches: `git branch --merged | xargs git branch -d`
- Stash including untracked files: `git stash -u`
Rebase onto main before opening a pull request to keep the history linear.
""",
    "notes/travel/packing-list.md": """# Packing list for a week of hiking

Rain jacket, two pairs of wool socks per day of walking, broken-in boots,
headlamp, first aid kit with blister plasters, water filter, power bank,
sunscreen, a paper map of the route in case the phone dies.
""",
    "notes/garden/tomatoes.md": """# Growing tomatoes on the balcony

Sow indoors in March, plant out after the last frost in mid May. Use a large
pot (at least 20 litres), stake the plants and remove side shoots every week.
Water deeply in the morning; uneven watering causes split fruit and blossom
end rot. Feed with tomato fertiliser once the first flowers appear.
""",
    "notes/family/kids-schedule.md": """# Kids weekly schedule

Monday: swimming lesson 17:00. Tuesday: piano at 16:30, bring the music book.
Wednesday: early pickup at 14:00. Thursday: football training.
Friday: library day, return the books. Dentist appointment on the 24th.
""",
}

EXTRA_PDFS = {
    "docs/car-manual.pdf": [
        [
            "Vehicle Owner's Manual - Maintenance",
            "",
            "Change the engine oil every 15,000 km or once a year.",
            "Check the tyre pressure monthly when the tyres are cold:",
            "2.4 bar front, 2.2 bar rear. Rotate the tyres every 10,000 km.",
        ],
        [
            "Warning lights",
            "",
            "A red battery symbol means the alternator is not charging.",
            "A yellow engine symbol means the engine control unit found a fault;",
            "drive gently and visit a workshop soon.",
        ],
    ],
}


def _window(
    title: str, size=(1200, 760), bg=(245, 245, 245)
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", size, (60, 70, 90))
    d = ImageDraw.Draw(im)
    d.rectangle((40, 40, size[0] - 40, size[1] - 40), fill=bg, outline=(80, 80, 80), width=2)
    d.rectangle((40, 40, size[0] - 40, 84), fill=(225, 225, 230))
    d.text((60, 50), title, font=font(22), fill=(20, 20, 20))
    return im, d


def wifi_settings() -> Image.Image:
    im, d = _window("Settings - Wi-Fi")
    d.text((80, 110), "Wi-Fi", font=font(36), fill=(20, 20, 20))
    nets = ["HomeNet-5G   (connected)", "HomeNet", "Cafe-Guest", "DIRECT-printer"]
    for i, n in enumerate(nets):
        y = 180 + 70 * i
        d.rectangle((80, y, 1100, y + 56), fill=(255, 255, 255), outline=(200, 200, 200))
        d.text((110, y + 14), n, font=font(24), fill=(30, 30, 30))
    d.text((80, 480), "Password", font=font(24), fill=(30, 30, 30))
    d.rectangle((80, 520, 700, 570), fill="white", outline=(120, 120, 120))
    d.text((95, 530), "**********", font=font(24), fill=(30, 30, 30))
    d.rectangle((740, 520, 900, 570), fill=(40, 110, 220))
    d.text((785, 530), "Join", font=font(24), fill="white")
    return im


def calendar_week() -> Image.Image:
    im, d = _window("Calendar - Week 41")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    events = {0: ("Team standup", 1), 1: ("Dentist 15:00", 4), 3: ("Project review", 2),
              4: ("Flight to Kyoto", 5)}  # fmt: skip
    for i, day in enumerate(days):
        x = 80 + 210 * i
        d.text((x + 70, 100), day, font=font(24), fill=(30, 30, 30))
        d.rectangle((x, 140, x + 200, 700), outline=(200, 200, 200))
        if i in events:
            t, slot = events[i]
            d.rectangle((x + 6, 150 + 90 * slot, x + 194, 220 + 90 * slot), fill=(120, 170, 240))
            d.text((x + 12, 170 + 90 * slot), t, font=font(18), fill=(10, 20, 60))
    return im


def network_diagram() -> Image.Image:
    im = Image.new("RGB", (1100, 700), "white")
    d = ImageDraw.Draw(im)
    nodes = {"Internet": (450, 40), "Router": (450, 260), "Laptop": (100, 520),
             "Phone": (450, 520), "Smart TV": (800, 520)}  # fmt: skip
    for name, (x, y) in nodes.items():
        d.rounded_rectangle((x, y, x + 200, y + 80), radius=12, fill=(230, 245, 230),
                            outline=(40, 120, 60), width=3)  # fmt: skip
        d.text((x + 30, y + 25), name, font=font(26), fill=(20, 60, 30))
    for a, b in [("Internet", "Router"), ("Router", "Laptop"), ("Router", "Phone"),
                 ("Router", "Smart TV")]:  # fmt: skip
        (ax, ay), (bx, by) = nodes[a], nodes[b]
        d.line((ax + 100, ay + 80, bx + 100, by), fill=(60, 60, 60), width=4)
    d.text((700, 300), "Wi-Fi 5 GHz", font=font(22), fill=(80, 80, 80))
    return im


def gantt_chart() -> Image.Image:
    im = Image.new("RGB", (1200, 600), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 20), "Website redesign - project timeline", font=font(30), fill=(20, 20, 20))
    tasks = [("Research", 0, 3), ("Wireframes", 2, 5), ("Visual design", 4, 8),
             ("Development", 6, 12), ("Testing", 10, 13), ("Launch", 13, 14)]  # fmt: skip
    for i, (t, s, e) in enumerate(tasks):
        y = 100 + 75 * i
        d.text((40, y + 10), t, font=font(20), fill=(30, 30, 30))
        d.rectangle((260 + 62 * s, y, 260 + 62 * e, y + 45), fill=(90, 140, 210))
    for w in range(15):
        d.text((260 + 62 * w, 560), f"W{w + 1}", font=font(14), fill=(90, 90, 90))
    return im


def mountain_lake() -> Image.Image:
    w, h = 1200, 800
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        t = y / h
        d.line((0, y, w, y), fill=(int(120 + 100 * t), int(170 + 60 * t), 235))
    d.polygon([(0, 520), (300, 160), (560, 520)], fill=(110, 110, 125))
    d.polygon([(380, 520), (720, 110), (1050, 520)], fill=(95, 100, 115))
    d.polygon([(640, 210), (720, 110), (800, 210)], fill=(250, 250, 250))
    d.polygon([(250, 220), (300, 160), (350, 220)], fill=(250, 250, 250))
    d.rectangle((0, 520, w, 620), fill=(40, 110, 60))
    d.rectangle((0, 620, w, h), fill=(50, 110, 170))
    rng = random.Random(5)
    for _ in range(60):
        x, y = rng.randint(0, w), rng.randint(630, h)
        d.line((x, y, x + rng.randint(20, 80), y), fill=(150, 190, 230), width=2)
    return im


def cactus_pot() -> Image.Image:
    im = Image.new("RGB", (800, 900), (245, 238, 225))
    d = ImageDraw.Draw(im)
    d.polygon([(250, 620), (550, 620), (510, 860), (290, 860)], fill=(190, 100, 60))
    d.rounded_rectangle((340, 230, 460, 640), radius=60, fill=(60, 150, 70))
    d.rounded_rectangle((220, 360, 300, 520), radius=40, fill=(60, 150, 70))
    d.rectangle((280, 470, 350, 510), fill=(60, 150, 70))
    d.rounded_rectangle((500, 300, 580, 470), radius=40, fill=(60, 150, 70))
    d.rectangle((450, 420, 520, 460), fill=(60, 150, 70))
    for y in range(260, 620, 40):
        for x in (360, 400, 440):
            d.line((x, y, x + 8, y - 8), fill=(240, 240, 220), width=2)
    return im


def spreadsheet() -> Image.Image:
    im, d = _window("expenses-2026.xlsx", size=(1200, 700), bg="white")
    cols = ["Date", "Category", "Description", "Amount"]
    rows = [("01-03", "Food", "Groceries", "84.20"), ("01-05", "Transport", "Train pass", "120.00"),
            ("01-09", "Home", "Electricity bill", "96.40"), ("01-12", "Food", "Restaurant", "45.00"),
            ("01-15", "Health", "Pharmacy", "18.75"), ("01-20", "Home", "Internet", "39.99")]  # fmt: skip
    for j, c in enumerate(cols):
        d.rectangle(
            (60 + 270 * j, 100, 330 + 270 * j, 140), fill=(220, 230, 220), outline=(170, 170, 170)
        )
        d.text((70 + 270 * j, 108), c, font=font(20), fill=(20, 20, 20))
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            y = 140 + 50 * i
            d.rectangle((60 + 270 * j, y, 330 + 270 * j, y + 50), outline=(200, 200, 200))
            d.text((70 + 270 * j, y + 14), v, font=font(20), fill=(30, 30, 30))
    return im


def bread_loaf(crust=(150, 90, 40)) -> Image.Image:
    im = Image.new("RGB", (1000, 750), (230, 220, 200))
    d = ImageDraw.Draw(im)
    d.ellipse((150, 560, 850, 660), fill=(150, 120, 90))  # board shadow
    d.ellipse((200, 200, 800, 600), fill=crust)
    for k in range(4):
        x = 330 + 110 * k
        d.arc((x - 60, 280, x + 60, 520), 250, 290, fill=(235, 210, 160), width=10)
    rng = random.Random(9)
    for _ in range(300):
        x, y = rng.randint(250, 750), rng.randint(240, 560)
        d.point((x, y), fill=(240, 230, 210))
    return im


EXTRA_IMAGES = {
    "screenshots/wifi-settings.png": wifi_settings,
    "screenshots/calendar.png": calendar_week,
    "screenshots/spreadsheet.png": spreadsheet,
    "diagrams/network-topology.png": network_diagram,
    "diagrams/project-timeline.png": gantt_chart,
    "photos/mountain-lake.jpg": mountain_lake,
    "photos/cactus.jpg": cactus_pot,
    "photos/bread-loaf.jpg": bread_loaf,
}

# ---------------------------------------------------------------- query images (not indexed)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def q_water_chart() -> Image.Image:
    return demo.bar_chart("Water usage per month (m3)", [9, 8, 9, 10, 12, 14, 15, 15, 12, 10, 9, 9],
                          MONTHS, (220, 120, 40))  # fmt: skip


def q_error_dialog() -> Image.Image:
    im = Image.new("RGB", (1100, 700), (40, 40, 60))
    d = ImageDraw.Draw(im)
    d.rectangle((250, 200, 850, 480), fill=(245, 245, 245), outline=(90, 90, 90), width=2)
    d.rectangle((250, 200, 850, 240), fill=(190, 50, 50))
    d.text((265, 208), "Warning", font=font(22), fill="white")
    d.text((300, 290), "Storage almost full", font=font(30), fill=(20, 20, 20))
    d.text((300, 340), "Only 120 MB left on this device.", font=font(20), fill=(40, 40, 40))
    d.text((300, 370), "Delete files to free up space.", font=font(20), fill=(40, 40, 40))
    return im


def q_signup_flow() -> Image.Image:
    im = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(im)
    steps = ["Open signup page", "Enter name + email", "Verify email link", "Account created"]
    for i, t in enumerate(steps):
        y = 40 + 165 * i
        d.rounded_rectangle((330, y, 670, y + 90), radius=12, fill=(255, 240, 220),
                            outline=(180, 110, 40), width=3)  # fmt: skip
        d.text((350, y + 30), t, font=font(22), fill=(60, 30, 10))
        if i < len(steps) - 1:
            d.line((500, y + 90, 500, y + 165), fill=(40, 40, 40), width=3)
    return im


def q_bicycle_blue() -> Image.Image:
    im = demo.bicycle()
    px = np.asarray(im).copy()
    red = (px[..., 0] > 150) & (px[..., 1] < 80)
    px[red] = (30, 70, 200)
    return Image.fromarray(px)


def q_beach() -> Image.Image:
    im = demo.sunset().transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return im.resize((1000, 667))


def q_whiteboard() -> Image.Image:
    im = Image.new("RGB", (1200, 800), (250, 250, 246))
    d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 1180, 780), outline=(150, 150, 150), width=8)
    d.text((80, 60), "Sprint goals", font=font(54), fill=(30, 120, 60))
    for i, t in enumerate(["- fix login bug", "- release v2.1", "- write docs"]):
        d.text((100, 200 + 90 * i), t, font=font(38), fill=(30, 30, 30))
    return im


def q_budget_pie() -> Image.Image:
    im = Image.new("RGB", (900, 650), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 20), "Team budget 2026", font=font(30), fill=(20, 20, 20))
    start = -90.0
    for pct, col in [(55, (200, 80, 80)), (25, (80, 160, 200)), (20, (220, 200, 80))]:
        d.pieslice(
            (150, 100, 650, 600), start, start + 3.6 * pct, fill=col, outline="white", width=3
        )
        start += 3.6 * pct
    return im


def q_bread() -> Image.Image:
    return bread_loaf((170, 105, 50)).rotate(8, fillcolor=(230, 220, 200)).resize((900, 675))


def q_router_photo() -> Image.Image:
    im = Image.new("RGB", (1000, 700), (225, 225, 230))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((200, 350, 800, 500), radius=20, fill=(30, 30, 35))
    for x in (280, 450, 620, 760):
        d.rectangle((x - 10, 120, x + 10, 360), fill=(40, 40, 45))
    for k in range(5):
        d.ellipse((260 + 60 * k, 410, 280 + 60 * k, 430), fill=(60, 220, 90))
    d.text((560, 405), "WiFi Router", font=font(26), fill=(200, 200, 200))
    return im


def q_noise() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8))


def q_gray() -> Image.Image:
    return Image.new("RGB", (800, 600), (128, 128, 128))


def q_stripes() -> Image.Image:
    im = Image.new("RGB", (900, 600), "white")
    d = ImageDraw.Draw(im)
    for i in range(0, 900, 60):
        d.rectangle((i, 0, i + 30, 600), fill=(250, 0, 200) if (i // 60) % 2 else (0, 230, 230))
    return im


def q_checker() -> Image.Image:
    a = (np.indices((600, 800)).sum(0) // 50) % 2
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


# ---------------------------------------------------------------- labelled queries
TEXT_QUERIES: list[tuple[str, list[str]]] = [
    ("how long should the dough proof", ["notes/sourdough.md", "photos/bread-loaf.jpg"]),
    ("sourdough starter feeding", ["notes/sourdough.md"]),
    ("photo of a loaf of bread", ["photos/bread-loaf.jpg", "notes/sourdough.md"]),
    ("temples to visit in Kyoto in autumn", ["notes/kyoto-trip.md"]),
    ("travel budget for Japan", ["notes/kyoto-trip.md"]),
    ("what did we decide about the marketing budget", ["notes/meetings/2026-03-budget.md"]),
    ("cannot borrow as mutable", ["notes/dev/rust-borrow-checker.md"]),
    (
        "free disk space docker build cache",
        ["notes/dev/docker-cheatsheet.md", "screenshots/error-dialog.png"],
    ),
    (
        "running out of disk space",
        ["screenshots/error-dialog.png", "notes/dev/docker-cheatsheet.md"],
    ),
    ("screenshot of an error about disk space", ["screenshots/error-dialog.png"]),
    ("how often to water succulents", ["notes/home/houseplants.md", "photos/cactus.jpg"]),
    ("picture of a cactus in a pot", ["photos/cactus.jpg", "notes/home/houseplants.md"]),
    ("interval training for a half marathon", ["notes/health/half-marathon-plan.md"]),
    ("knee pain from running too much", ["notes/health/half-marathon-plan.md"]),
    (
        "documents needed for the tax return",
        ["notes/finance/tax-checklist.md", "notes/finance/kakutei-shinkoku.md"],
    ),
    ("確定申告の期限", ["notes/finance/kakutei-shinkoku.md", "notes/finance/tax-checklist.md"]),
    (
        "medical expense deduction",
        ["notes/finance/kakutei-shinkoku.md", "notes/finance/tax-checklist.md"],
    ),
    ("how to build a new habit", ["notes/reading/habits.md"]),
    ("raspberry pi weather station", ["notes/ideas.txt"]),
    ("birthday present ideas", ["notes/ideas.txt"]),
    ("chickpea coconut curry recipe", ["notes/recipes/vegetable-curry.md"]),
    ("reset the wifi password", ["docs/home-network-manual.pdf", "screenshots/wifi-settings.png"]),
    ("router admin page address", ["docs/home-network-manual.pdf"]),
    ("why is my internet slow", ["docs/home-network-manual.pdf"]),
    (
        "diagram of the home network",
        ["diagrams/network-topology.png", "docs/home-network-manual.pdf"],
    ),
    ("how many vacation days do new employees get", ["docs/employee-handbook.pdf"]),
    ("remote work policy", ["docs/employee-handbook.pdf"]),
    ("solar panel yearly production", ["docs/solar-panel-report.pdf"]),
    (
        "chart of monthly electricity use",
        ["screenshots/electricity-usage.png", "docs/solar-panel-report.pdf"],
    ),
    ("python traceback missing module", ["screenshots/terminal.png"]),
    ("login flow diagram with two factor", ["diagrams/login-flow.png"]),
    ("pie chart of household spending", ["diagrams/spending-pie.png"]),
    ("sunset at the beach", ["photos/sunset-beach.jpg"]),
    ("red bicycle", ["photos/red-bicycle.jpg"]),
    ("whiteboard with the quarterly roadmap", ["photos/whiteboard.jpg"]),
    ("undo the last git commit", ["notes/dev/git-tips.md"]),
    ("what to pack for a hiking trip", ["notes/travel/packing-list.md"]),
    ("tomato plants splitting", ["notes/garden/tomatoes.md"]),
    ("when is the piano lesson", ["notes/family/kids-schedule.md", "screenshots/calendar.png"]),
    ("dentist appointment", ["notes/family/kids-schedule.md", "screenshots/calendar.png"]),
    ("correct tyre pressure", ["docs/car-manual.pdf"]),
    ("what does the engine warning light mean", ["docs/car-manual.pdf"]),
    ("wifi network list in settings", ["screenshots/wifi-settings.png"]),
    ("calendar screenshot with meetings", ["screenshots/calendar.png"]),
    ("spreadsheet of monthly expenses", ["screenshots/spreadsheet.png"]),
    ("project timeline gantt chart", ["diagrams/project-timeline.png"]),
    ("snowy mountains and a lake", ["photos/mountain-lake.jpg"]),
]

# Clearly unrelated to everything in the corpus.
NEGATIVE_QUERIES: list[str] = [
    "quantum chromodynamics lattice gauge theory",
    "how to train a parrot to talk",
    "medieval castle siege weapons",
    "best price for a used tractor",
    "symptoms of feline diabetes",
    "rules of cricket lbw",
    "jazz saxophone improvisation scales",
    "deep sea anglerfish bioluminescence",
    "knitting pattern for a wool scarf",
    "history of the ottoman empire",
    "volcano eruption in iceland",
    "how to tie a bowline knot",
    "opera singers of the 19th century",
    "submarine sonar operation",
    "wedding speech for my brother",
    "asdf qwer zxcv",
    "lorem ipsum dolor sit amet",
    "kjhsdf8723 xqv",
    "zzzzzzzz",
    "blorp fizzwhack glimmerstone",
    "ほげほげ ふがふが",
    "the",
    "???",
    "a",
    # Short or meaningless queries a first-time user might type.
    "hi",
    "test test",
    "1234",
    "ok",
    "!!!",
    "hmm",
    "qwerty",
    "untitled",
    "あ",
    "えーと",
    "こんにちは",
    "なし",
]

# (query image factory, relevant files)
IMAGE_QUERIES: list[tuple[str, object, list[str]]] = [
    ("water-chart", q_water_chart, ["screenshots/electricity-usage.png"]),
    (
        "storage-warning",
        q_error_dialog,
        ["screenshots/error-dialog.png", "notes/dev/docker-cheatsheet.md"],
    ),
    ("signup-flow", q_signup_flow, ["diagrams/login-flow.png"]),
    ("blue-bicycle", q_bicycle_blue, ["photos/red-bicycle.jpg"]),
    ("beach", q_beach, ["photos/sunset-beach.jpg"]),
    ("sprint-whiteboard", q_whiteboard, ["photos/whiteboard.jpg"]),
    ("budget-pie", q_budget_pie, ["diagrams/spending-pie.png"]),
    ("bread", q_bread, ["photos/bread-loaf.jpg", "notes/sourdough.md"]),
    ("router", q_router_photo, ["docs/home-network-manual.pdf", "diagrams/network-topology.png"]),
]
NEGATIVE_IMAGE_QUERIES = [("noise", q_noise), ("gray", q_gray), ("stripes", q_stripes),
                          ("checker", q_checker)]  # fmt: skip


# ---------------------------------------------------------------- run
def build_corpus(root: Path) -> None:
    for rel, text in {**demo.NOTES, **EXTRA_NOTES}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for rel, pages in {**demo.PDFS, **EXTRA_PDFS}.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        demo.write_pdf(p, pages)
    months = MONTHS
    images = {
        "screenshots/error-dialog.png": demo.error_dialog,
        "screenshots/terminal.png": demo.terminal_traceback,
        "screenshots/electricity-usage.png": lambda: demo.bar_chart(
            "Monthly electricity usage (kWh)",
            [420, 390, 350, 300, 260, 240, 250, 255, 280, 330, 380, 430],
            months,
            (60, 120, 200),
        ),
        "diagrams/login-flow.png": demo.flowchart,
        "diagrams/spending-pie.png": demo.pie_chart,
        "photos/sunset-beach.jpg": demo.sunset,
        "photos/red-bicycle.jpg": demo.bicycle,
        "photos/whiteboard.jpg": demo.whiteboard,
        **EXTRA_IMAGES,
    }
    for rel, make in images.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        im = make()
        im.save(p, quality=90) if p.suffix == ".jpg" else im.save(p)


def file_scores(store, qvec, root: Path, query_kind: str, budget: int) -> dict:
    """Best chunk per file, as the default search output shows it:
    {relative path: {kind, score, level, weak, legacy_weak}}, best first.
    `level` is the query's reference level for that type; `legacy_weak`
    applies the old median rule instead of the reference level."""
    _, mat, paths, kinds = store.load_vectors()
    s = mat @ np.asarray(qvec, np.float32)
    kinds_a = np.asarray(kinds)
    cut = thresholds(qvec, kinds, query_kind, budget)
    legacy = {}
    for k in set(kinds):
        sel = kinds_a == k
        t = min_score(query_kind, k, budget)
        if sel.sum() >= LEGACY_MIN_ROWS:
            t = max(t, float(np.median(s[sel])) + LEGACY_MARGIN)
        legacy[k] = t
    out: dict[str, dict] = {}
    for i in np.argsort(-s, kind="stable"):
        rel = Path(paths[i]).relative_to(root).as_posix()
        if rel in out:
            continue
        k = kinds[i]
        out[rel] = {
            "kind": k,
            "score": float(s[i]),
            "level": reference_level(qvec, k, budget),
            "weak": bool(s[i] < cut[k]),
            "legacy_weak": bool(s[i] < legacy[k]),
        }
    return out


def collect(store, embedder, root: Path, budget: int, qsets) -> list[dict]:
    text_q, neg_q, image_q, neg_image_q = qsets
    pairs: list[dict] = []

    def add(qtype: str, qname: str, qvec, relevant: set[str] | None) -> None:
        shown = 0
        for rel, f in file_scores(store, qvec, root, qtype, budget).items():
            label = "neg" if relevant is None else ("rel" if rel in relevant else "irr")
            top10 = not f["weak"] and shown < 10
            shown += not f["weak"]
            pairs.append({"pair": f"{qtype}->{f['kind']}", "query": qname, "file": rel,
                          "label": label, "score": round(f["score"], 4),
                          "excess": round(f["score"] - f["level"], 4),
                          "weak": f["weak"], "legacy_weak": f["legacy_weak"],
                          "top10": top10})  # fmt: skip

    for q, relevant in text_q:
        add("text", q, embedder.embed_query(q), set(relevant))
    for q in neg_q:
        add("text", q, embedder.embed_query(q), None)
    embedder.image_budget = budget
    for name, make, relevant in image_q:
        add("image", name, embedder.embed_image(make()), set(relevant))
    for name, make in neg_image_q:
        add("image", name, embedder.embed_image(make()), None)
    return pairs


def _cap(t: float, neg: list[float]) -> float:
    """No higher than needed to drop every result of the unrelated queries:
    the best unrelated value, rounded up to the next 0.01."""
    return min(t, math.ceil(max(neg) * 100) / 100) if neg else t


def suggest(rel: list[float], neg: list[float], min_recall: float = 0.95) -> float:
    """The minimum score: the highest cutoff (in steps of 0.01) that keeps at
    least `min_recall` of the relevant pairs, capped by _cap."""
    best = math.floor(min(rel) * 100) / 100
    t = best
    while True:
        t2 = round(t + 0.01, 2)
        if sum(s >= t2 for s in rel) / len(rel) < min_recall:
            break
        t = t2
    return round(_cap(t, neg), 2)


def suggest_margin(ps: list[dict], cutoff: float, min_recall: float = 0.95) -> float:
    """The margin above the reference level: the highest value (in steps of
    0.01) that keeps `min_recall` of the relevant pairs, or all of those that
    pass the minimum score if that is fewer. Unlike the minimum score it is not
    capped by the unrelated queries of this corpus: on held-out data a margin
    that only just drops them let meaningless queries through."""
    rel = [p for p in ps if p["label"] == "rel"]
    passing = [p["excess"] for p in rel if p["score"] >= cutoff]
    if not passing:
        return 0.0
    need = min(math.ceil(min_recall * len(rel)), len(passing))
    m = 0.0
    while sum(e >= round(m + 0.01, 2) for e in passing) >= need:
        m = round(m + 0.01, 2)
    return m


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals)
    return {
        "n": len(vals),
        "min": round(float(a.min()), 3),
        "p05": round(float(np.quantile(a, 0.05)), 3),
        "median": round(float(np.median(a)), 3),
        "p95": round(float(np.quantile(a, 0.95)), 3),
        "max": round(float(a.max()), 3),
    }


def _pr(ps: list[dict], passes) -> tuple[float | None, float | None, int]:
    rel = [p for p in ps if p["label"] == "rel"]
    tp = sum(passes(p) for p in rel)
    fp = sum(passes(p) for p in ps if p["label"] != "rel")
    neg = sum(passes(p) for p in ps if p["label"] == "neg")
    recall = round(tp / len(rel), 3) if rel else None
    precision = round(tp / (tp + fp), 3) if tp + fp else None
    return recall, precision, neg


def evaluate(pairs: list[dict], budget: int) -> dict:
    res: dict[str, dict] = {}
    for key in sorted({p["pair"] for p in pairs}):
        ps = [p for p in pairs if p["pair"] == key]
        qk, rk = key.split("->")
        t = min_score(qk, rk, budget=budget)
        rel = [p["score"] for p in ps if p["label"] == "rel"]
        neg = [p["score"] for p in ps if p["label"] == "neg"]
        r0, p0, n0 = _pr(ps, lambda p: True)
        r1, p1, n1 = _pr(ps, lambda p, t=t: p["score"] >= t)
        r2, p2, n2 = _pr(ps, lambda p: not p["weak"])
        r3, p3, n3 = _pr(ps, lambda p: not p["legacy_weak"])
        res[key] = {
            "threshold": t,
            "suggested": suggest(rel, neg) if rel else None,
            "margin": min_margin(qk, rk),
            "suggested_margin": suggest_margin(ps, t) if rel else None,
            "relevant": summarize(rel),
            "relevant_excess": summarize([p["excess"] for p in ps if p["label"] == "rel"]),
            "irrelevant_same_query": summarize([p["score"] for p in ps if p["label"] == "irr"]),
            "unrelated_queries": summarize(neg),
            "unrelated_excess": summarize([p["excess"] for p in ps if p["label"] == "neg"]),
            "no_cutoff": {"recall": r0, "precision": p0, "unrelated_shown": n0},
            "min_score_only": {"recall": r1, "precision": p1, "unrelated_shown": n1},
            "full_rule": {"recall": r2, "precision": p2, "unrelated_shown": n2},
            "legacy_median_rule": {"recall": r3, "precision": p3, "unrelated_shown": n3},
        }
    return res


def per_query(pairs: list[dict]) -> dict:
    """Per-query view of the full rule (and the legacy rule): how many
    unrelated queries still show anything, how many relevant files are hidden
    or fall outside the top 10, and how many unlabelled files a relevant query
    shows on average."""
    out = {}
    for name, weak in (("full_rule", "weak"), ("legacy_median_rule", "legacy_weak")):
        neg_q = sorted({p["query"] for p in pairs if p["label"] == "neg"})
        neg_any = sorted({p["query"] for p in pairs if p["label"] == "neg" and not p[weak]})
        rel = [p for p in pairs if p["label"] == "rel"]
        hidden = [f"{p['query']} -> {p['file']} ({p['score']:.3f})" for p in rel if p[weak]]
        rel_q = {p["query"] for p in rel}
        other = sum(1 for p in pairs if p["query"] in rel_q and p["label"] == "irr" and not p[weak])
        out[name] = {
            "relevant_shown": f"{len(rel) - len(hidden)}/{len(rel)}",
            "unrelated_queries_with_results": f"{len(neg_any)}/{len(neg_q)}",
            "unrelated_with_results": neg_any,
            "relevant_hidden": hidden,
            "unlabelled_shown_per_query": round(other / max(len(rel_q), 1), 2),
        }
    out["full_rule"]["relevant_outside_top10"] = [
        f"{p['query']} -> {p['file']}"
        for p in pairs
        if p["label"] == "rel" and not p["weak"] and not p["top10"]
    ]
    return out


def _print_pairs(res: dict) -> None:
    print(f"{'pair':<14}{'min':>5}{'sugg':>6}{'marg':>6}{'sugg':>6}{'rel min/med':>13}"
          f"{'unrel max':>10}{'   R/P min score':>17}{'   R/P full rule':>17}"
          f"{'   R/P old rule':>17}")  # fmt: skip
    for key, r in res.items():
        rel, neg = r["relevant"], r["unrelated_queries"]

        def rp(d: dict) -> str:
            f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
            return f"{f(d['recall'])}/{f(d['precision'])} ({d['unrelated_shown']})"

        print(
            f"{key:<14}{r['threshold']:>5.2f}{r['suggested'] or math.nan:>6.2f}"
            f"{r['margin']:>6.2f}{r['suggested_margin'] or math.nan:>6.2f}"
            f"{rel.get('min', math.nan):>7.3f}{rel.get('median', math.nan):>6.3f}"
            f"{neg.get('max', math.nan):>10.3f}"
            f"{rp(r['min_score_only']):>17}{rp(r['full_rule']):>17}"
            f"{rp(r['legacy_median_rule']):>17}"
        )


def _print_summary(sm: dict) -> None:
    for name in ("full_rule", "legacy_median_rule"):
        s = sm[name]
        print(f"  {name}: relevant shown {s['relevant_shown']}, unrelated queries with results "
              f"{s['unrelated_queries_with_results']}, unlabelled files shown per relevant query "
              f"{s['unlabelled_shown_per_query']}")  # fmt: skip
        print(f"    unrelated with results: {s['unrelated_with_results']}")
        print(f"    relevant hidden: {s['relevant_hidden']}")
    print(f"  relevant outside the top 10: {sm['full_rule']['relevant_outside_top10']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--budgets", default="140,70,280")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--work", type=Path, help="where to build the corpora and indexes")
    ap.add_argument("--json", type=Path, default=ROOT / "docs" / "calibration-results.json")
    ap.add_argument("--pairs", type=Path, help="also write every (query, file, score) row here")
    ap.add_argument("--no-heldout", action="store_true", help="skip the held-out corpora")
    args = ap.parse_args()

    from anygrep.config import model_dir
    from anygrep.embedder import OnnxEmbedder
    from anygrep.indexer import IndexOptions, run_index
    from anygrep.store import Store

    if args.work is None:
        import tempfile

        args.work = Path(tempfile.mkdtemp(prefix="anygrep-calibration-"))
    root = (args.work / "corpus").resolve()
    build_corpus(root)
    embedder = OnnxEmbedder(model_dir(), threads=args.threads)
    store = Store(args.work / "index.sqlite")
    rule = {
        "min_scores": {k: dict(v) for k, v in MIN_SCORES.items()},
        "min_margins": dict(MIN_MARGINS),
    }
    queries = {
        "text": len(TEXT_QUERIES),
        "text_unrelated": len(NEGATIVE_QUERIES),
        "image": len(IMAGE_QUERIES),
        "image_unrelated": len(NEGATIVE_IMAGE_QUERIES),
    }
    qsets = (TEXT_QUERIES, NEGATIVE_QUERIES, IMAGE_QUERIES, NEGATIVE_IMAGE_QUERIES)
    report: dict = {"rule": rule, "queries": queries, "budgets": {}, "heldout": {}}
    all_pairs: dict[str, list[dict]] = {}
    for budget in [int(b) for b in args.budgets.split(",")]:
        embedder.image_budget = budget
        run_index(store, [root], IndexOptions(image_budget=budget, quiet=True), lambda v: embedder)
        pairs = collect(store, embedder, root, budget, qsets)
        all_pairs[str(budget)] = pairs
        res = evaluate(pairs, budget)
        report["budgets"][str(budget)] = {"pairs": res, "summary": per_query(pairs)}
        print(f"\n== calibration corpus, image budget {budget} ==")
        _print_pairs(res)
        _print_summary(report["budgets"][str(budget)]["summary"])
    store.close()

    if not args.no_heldout:
        heldout = _load_script("heldout_corpus")
        budget = 140
        embedder.image_budget = budget
        for name in heldout.CORPORA:
            hroot = (args.work / "heldout" / name).resolve()
            hroot.mkdir(parents=True, exist_ok=True)
            heldout.build(hroot, name)
            hstore = Store(args.work / f"heldout-{name}.sqlite")
            opts = IndexOptions(image_budget=budget, quiet=True)
            run_index(hstore, [hroot], opts, lambda v: embedder)
            pairs = collect(hstore, embedder, hroot, budget, heldout.QSETS[name])
            hstore.close()
            all_pairs[f"heldout-{name}"] = pairs
            res = evaluate(pairs, budget)
            sm = per_query(pairs)
            report["heldout"][name] = {"pairs": res, "summary": sm}
            print(f"\n== held-out corpus '{name}', image budget {budget} ==")
            _print_pairs(res)
            _print_summary(sm)

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"\nWrote {args.json}  (R/P: recall/precision, in brackets: results of unrelated queries)"
    )
    if args.pairs:
        args.pairs.write_text(json.dumps(all_pairs, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
