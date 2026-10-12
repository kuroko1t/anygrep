"""Held-out corpora for checking the relevance cutoff on data it was not tuned on.

scripts/calibrate_thresholds.py picks the cutoffs on its own corpus and
queries, then reports how the same rule does on these corpora. Everything is
synthetic (written or drawn here), on topics that the calibration corpus does
not cover:

  mixed      notes, README-style project notes with code, Japanese notes,
             PDFs and 14 pictures (weather apps, charts, a receipt, a chess
             board, a timetable ...), with English, Japanese and cross-lingual
             queries, off-topic questions and short or meaningless queries.
  recipes    a folder of 10 recipes (one topic).
  sourdough  8 notes about sourdough baking and 2 unrelated notes, the case
             where most files of a folder match the query.
  akiya      6 Japanese notes about vacant-house policy (one topic).

Japanese text in pictures needs a CJK font (Noto Sans CJK); without one the
characters are drawn as boxes and the numbers change a little.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "make_demo_corpus", ROOT / "scripts" / "make_demo_corpus.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo = _load_demo()
F = demo.font


def J(size):
    """A font with Japanese glyphs, if one is installed."""
    for name in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "NotoSansCJK-Regular.ttc",
        "NotoSansJP-Regular.otf",
        "msgothic.ttc",
        "Hiragino Sans GB.ttc",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return F(size)


# ------------------------------------------------------------------ mixed: texts
NOTES = {
    "notes/aquarium.md": """# Freshwater aquarium care

Cycle a new tank for 4-6 weeks before adding fish so bacteria can turn ammonia
into nitrite and then nitrate. Change 25% of the water every week, using a
dechlorinator. Neon tetras like soft water around 24 C and should be kept in a
group of at least six. Rinse the filter sponge in old tank water, never tap water.
""",
    "notes/miso-soup.txt": """味噌汁の作り方（2人分）
昆布を水に30分つけてから火にかけ、沸騰直前に取り出す。かつお節を入れて2分煮て濾す。
豆腐とわかめを入れ、火を止めてから味噌を大さじ2溶き入れる。味噌を入れたら沸騰させない。
仕上げに刻んだ長ねぎを散らす。
""",
    "notes/meeting-2026-09-12.md": """# 定例会議 議事録 2026-09-12

出席: 佐藤、田中、鈴木
- オフィス移転: 10月末に新オフィスへ移る。引越し業者は3社から見積を取る（担当: 田中）。
- 予算は200万円以内。ネット回線の工事は2週間前までに申し込む。
- 次回は9月26日。
""",
    "notes/todo.txt": """call plumber re leaking kitchen tap
renew passport (expires Jan)
cancel gym membership before the 30th
buy printer ink, black XL
""",
}
NOTES.update(
    {
        "projects/imgshrink/README.md": """# imgshrink

Batch-resize and recompress photos from the command line.

    pip install imgshrink
    imgshrink ~/Pictures/trip --max-side 2048 --quality 82 --out ~/Pictures/trip-small

Options: `--keep-exif` keeps camera metadata, `--webp` writes WebP instead of
JPEG, `--dry-run` only prints what would change. Uses all CPU cores by default;
limit with `-j 2`.
""",
        "projects/kakeibo/README.md": """# kakeibo - 家計簿アプリ

レシートの金額を手入力して、月ごとの食費・日用品・交通費を集計するWebアプリ。
グラフで前月と比べられる。データはブラウザのIndexedDBに保存し、サーバーには送らない。

    npm install
    npm run dev

TODO: CSVエクスポート、カテゴリの並べ替え。
""",
        "projects/backup.sh.md": """# Nightly backup

```sh
#!/bin/sh
set -eu
restic -r /mnt/nas/restic backup ~/Documents ~/Pictures --exclude-caches
restic -r /mnt/nas/restic forget --keep-daily 7 --keep-weekly 4 --prune
```

Runs from cron at 02:30. Check `restic snapshots` once a month and test a
restore of a single file.
""",
        "notes/language-exchange.md": """# 言語交換メモ

毎週土曜10時にオンラインで1時間。前半30分は英語、後半30分は日本語。
話題リスト: 好きな映画、週末の過ごし方、地元の食べ物。
間違いはその場で直さず、最後にまとめてチャットで送る。
""",
        "notes/car-insurance.md": """# Car insurance renewal

Current policy ends 31 March. Quotes: 612 (same insurer), 548 (online only, no
courtesy car), 590 (includes windscreen cover). Excess 300 on all three.
No-claims bonus: 6 years. Decide by 15 March and cancel auto-renewal.
""",
        "notes/kanji-study.md": """# 漢字の勉強計画

漢検2級の試験は11月。1日20字ずつ復習し、週末に書き取りテストをする。
間違えた字はノートの最後のページにまとめる。部首ごとに覚えると忘れにくい。
""",
    }
)

SYN_PDFS = {
    "pdf/homebrew-beer.pdf": [
        [
            "Homebrewing a pale ale",
            "",
            "Mash the crushed malt at 66 C for 60 minutes, then sparge.",
            "Boil the wort for an hour; add bittering hops at the start and aroma hops",
            "in the last ten minutes. Cool quickly to 20 C before pitching the yeast.",
        ],
        [
            "Fermentation and bottling",
            "",
            "Ferment at 18-20 C for two weeks.",
            "Sanitize every bottle. Add 6 g of priming sugar per litre for carbonation",
            "and leave the bottles warm for two more weeks before chilling.",
        ],
    ],
    "pdf/lease-agreement.pdf": [
        [
            "Residential Lease Agreement",
            "",
            "Rent of 1,150 EUR is due on the 1st of each month.",
            "The security deposit equals two months of rent and is returned within",
            "30 days after the tenant moves out, minus the cost of any damage.",
        ],
        [
            "Termination and house rules",
            "",
            "Either party may end the lease with 60 days written notice.",
            "Pets are not allowed without written permission from the landlord.",
            "Quiet hours are from 22:00 to 07:00.",
        ],
    ],
}

SYN_PDFS["pdf/cycling-strategy.pdf"] = [
    [
        "City cycling strategy 2026-2030",
        "",
        "This plan sets out how the city will double the share",
        "of trips made by bicycle by 2030. It covers protected lanes, parking, safety",
        "campaigns and how progress will be measured.",
    ],
    [
        "1. Protected lanes",
        "",
        "Build 45 km of lanes separated from traffic by kerbs or",
        "planters along the main radial roads. Junctions get separate signal phases",
        "for cyclists. Painted lanes on busy roads will be upgraded first.",
    ],
    [
        "2. Bicycle parking",
        "",
        "Add 3,000 stands near stations, schools and shops, and",
        "secure covered parking at the four largest stations. Residential streets can",
        "request on-street bike hangars.",
    ],
    [
        "3. Safety",
        "",
        "Lower the speed limit to 30 km/h in residential areas. Run",
        "school cycling courses for every pupil aged 10. Publish collision data monthly.",
    ],
    [
        "4. Monitoring",
        "",
        "Automatic counters at 25 sites, a yearly travel survey and",
        "a public dashboard. The council reviews the plan every two years.",
    ],
]


# ------------------------------------------------------------------ images
def _win(title, size=(1200, 760), jp=False):
    im = Image.new("RGB", size, (50, 55, 70))
    d = ImageDraw.Draw(im)
    d.rectangle((30, 30, size[0] - 30, size[1] - 30), fill=(250, 250, 252))
    d.rectangle((30, 30, size[0] - 30, 76), fill=(220, 224, 232))
    d.text((50, 40), title, font=(J if jp else F)(22), fill=(20, 20, 20))
    return im, d


def _sun(d, x, y, r=26):
    d.ellipse((x - r, y - r, x + r, y + r), fill=(250, 200, 40))


def _cloud(d, x, y, rain=False):
    for dx, dy, r in [(-22, 4, 20), (0, -8, 26), (24, 4, 20)]:
        d.ellipse((x + dx - r, y + dy - r, x + dx + r, y + dy + r), fill=(170, 175, 190))
    if rain:
        for k in range(4):
            d.line((x - 24 + 16 * k, y + 28, x - 30 + 16 * k, y + 46), fill=(50, 110, 220), width=3)


def weather_app(city="Berlin", seed=1, accent=(40, 110, 220)):
    im, d = _win(f"Weather - {city}")
    d.text((70, 100), city, font=F(44), fill=(20, 20, 20))
    d.text((70, 160), "18 C  Partly cloudy", font=F(30), fill=accent)
    rng = random.Random(seed)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i, day in enumerate(days):
        x = 90 + 150 * i
        d.text((x, 280), day, font=F(24), fill=(40, 40, 40))
        k = rng.randint(0, 2)
        (_sun(d, x + 25, 360) if k == 0 else _cloud(d, x + 25, 360, rain=k == 2))
        d.text(
            (x, 430), f"{rng.randint(12, 22)} / {rng.randint(4, 11)}", font=F(22), fill=(40, 40, 40)
        )
    d.text(
        (70, 540), "Precipitation 40%   Wind 12 km/h   Humidity 65%", font=F(24), fill=(60, 60, 60)
    )
    return im


def tenki_jp():
    im, d = _win("天気予報 - 東京", jp=True)
    d.text((70, 100), "東京都 千代田区", font=J(40), fill=(20, 20, 20))
    days = ["月", "火", "水", "木", "金"]
    w = ["晴れ", "くもり", "雨", "晴れ", "くもり時々雨"]
    for i in range(5):
        x = 90 + 210 * i
        d.text((x, 200), f"{10 + i}日({days[i]})", font=J(26), fill=(30, 30, 30))
        (_sun(d, x + 50, 290) if w[i] == "晴れ" else _cloud(d, x + 50, 290, rain="雨" in w[i]))
        d.text((x, 360), w[i], font=J(24), fill=(30, 30, 30))
        d.text((x, 410), f"最高 {24 - i}℃", font=J(22), fill=(200, 60, 40))
        d.text((x, 450), f"降水確率 {[0, 30, 80, 10, 50][i]}%", font=J(22), fill=(40, 90, 200))
    return im


def rainfall_chart():
    return demo.bar_chart(
        "Average annual rainfall by city (mm)",
        [1530, 590, 1270, 640, 2400, 820],
        ["Tokyo", "Berlin", "Seattle", "Madrid", "Mumbai", "Sydney"],
        (70, 150, 120),
    )


def line_chart(title, vals, color, labels):
    im = Image.new("RGB", (1100, 650), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 25), title, font=F(30), fill=(20, 20, 20))
    x0, y0, x1, y1 = 100, 100, 1050, 560
    d.line((x0, y1, x1, y1), fill="black", width=2)
    d.line((x0, y0, x0, y1), fill="black", width=2)
    lo, hi = min(vals), max(vals)
    pts = [
        (x0 + (x1 - x0) * i / (len(vals) - 1), y1 - 20 - (y1 - y0 - 40) * (v - lo) / (hi - lo))
        for i, v in enumerate(vals)
    ]
    d.line(pts, fill=color, width=4)
    for i, lab in enumerate(labels):
        d.text(
            (x0 + (x1 - x0) * i / (len(labels) - 1) - 15, y1 + 12),
            lab,
            font=F(16),
            fill=(60, 60, 60),
        )
    return im


def stock_chart():
    rng = random.Random(3)
    v, vals = 100.0, []
    for _ in range(60):
        v *= 1 + rng.gauss(0.004, 0.03)
        vals.append(v)
    im = line_chart(
        "ACME Corp share price 2025 (USD)",
        vals,
        (200, 50, 50),
        ["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
    )
    return im


def scatter():
    im = Image.new("RGB", (1000, 700), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 20), "Height vs weight of 200 adults", font=F(30), fill=(20, 20, 20))
    d.line((100, 620, 960, 620), fill="black", width=2)
    d.line((100, 80, 100, 620), fill="black", width=2)
    d.text((480, 640), "Height (cm)", font=F(18), fill="black")
    d.text((10, 330), "Weight (kg)", font=F(18), fill="black")
    rng = random.Random(7)
    for _ in range(200):
        h = rng.gauss(170, 9)
        w = 0.9 * (h - 100) + rng.gauss(0, 8)
        x, y = 100 + (h - 140) * 14, 620 - (w - 35) * 7
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(60, 100, 200))
    return im


def org_chart(names=None, color=(220, 230, 250)):
    names = names or {
        "CEO": "Maria Lopez",
        "CTO": "Ken Ito",
        "CFO": "Anna Berg",
        "COO": "Sam Okafor",
    }
    im = Image.new("RGB", (1200, 720), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 20), "Company organization chart", font=F(30), fill=(20, 20, 20))

    def box(x, y, t1, t2):
        d.rectangle((x, y, x + 240, y + 80), fill=color, outline=(40, 60, 120), width=2)
        d.text((x + 15, y + 10), t1, font=F(22), fill=(20, 20, 60))
        d.text((x + 15, y + 44), t2, font=F(18), fill=(60, 60, 60))

    keys = list(names)
    box(480, 90, keys[0], names[keys[0]])
    for i, k in enumerate(keys[1:]):
        x = 120 + 360 * i
        d.line((600, 170, x + 120, 300), fill=(80, 80, 80), width=3)
        box(x, 300, k, names[k])
        for j, team in enumerate(["Team A", "Team B"]):
            tx = x - 40 + 170 * j
            d.line((x + 120, 380, tx + 80, 500), fill=(120, 120, 120), width=2)
            d.rectangle((tx, 500, tx + 160, 560), fill=(240, 240, 240), outline=(120, 120, 120))
            d.text((tx + 30, 515), team, font=F(18), fill=(40, 40, 40))
    return im


def sequence_diagram():
    im = Image.new("RGB", (1100, 760), "white")
    d = ImageDraw.Draw(im)
    actors = {"Browser": 150, "API server": 550, "Database": 950}
    for n, x in actors.items():
        d.rectangle((x - 90, 30, x + 90, 90), fill=(255, 245, 210), outline=(120, 100, 40), width=2)
        d.text((x - 70, 48), n, font=F(22), fill=(40, 30, 10))
        for y in range(90, 740, 16):
            d.line((x, y, x, y + 8), fill=(120, 120, 120), width=2)
    msgs = [
        ("Browser", "API server", "GET /orders"),
        ("API server", "Database", "SELECT * FROM orders"),
        ("Database", "API server", "rows"),
        ("API server", "Browser", "200 OK (JSON)"),
    ]
    for i, (a, b, t) in enumerate(msgs):
        y = 160 + 130 * i
        xa, xb = actors[a], actors[b]
        d.line((xa, y, xb, y), fill=(30, 30, 30), width=3)
        s = 1 if xb > xa else -1
        d.polygon([(xb, y), (xb - 14 * s, y - 8), (xb - 14 * s, y + 8)], fill=(30, 30, 30))
        d.text((min(xa, xb) + 30, y - 32), t, font=F(20), fill=(20, 20, 120))
    return im


def music_player():
    im = Image.new("RGB", (900, 1200), (24, 24, 28))
    d = ImageDraw.Draw(im)
    d.rectangle((150, 120, 750, 720), fill=(180, 70, 120))
    d.ellipse((300, 270, 600, 570), fill=(30, 30, 30))
    d.ellipse((420, 390, 480, 450), fill=(200, 200, 200))
    d.text((150, 760), "Now Playing", font=F(24), fill=(170, 170, 170))
    d.text((150, 800), "Midnight Train - The Lanterns", font=F(34), fill="white")
    d.rectangle((150, 880, 750, 888), fill=(80, 80, 80))
    d.rectangle((150, 880, 420, 888), fill=(30, 215, 96))
    d.text((150, 900), "1:42", font=F(18), fill=(170, 170, 170))
    d.text((700, 900), "3:58", font=F(18), fill=(170, 170, 170))
    d.ellipse((400, 960, 500, 1060), fill="white")
    d.rectangle((432, 985, 444, 1035), fill=(24, 24, 28))
    d.rectangle((456, 985, 468, 1035), fill=(24, 24, 28))
    d.polygon([(300, 1010), (340, 985), (340, 1035)], fill="white")
    d.polygon([(600, 1010), (560, 985), (560, 1035)], fill="white")
    return im


def chessboard(light=(240, 217, 181), dark=(181, 136, 99), seed=2):
    im = Image.new("RGB", (800, 800), "white")
    d = ImageDraw.Draw(im)
    s = 100
    for r in range(8):
        for c in range(8):
            d.rectangle(
                (c * s, r * s, c * s + s, r * s + s), fill=light if (r + c) % 2 == 0 else dark
            )
    rng = random.Random(seed)
    letters = "KQRRBBNNPPPPPP"
    for side, col in ((0, (250, 250, 250)), (1, (20, 20, 20))):
        for ch in letters:
            r, c = rng.randint(0, 7), rng.randint(0, 7)
            x, y = c * s + 50, r * s + 50
            d.ellipse((x - 34, y - 34, x + 34, y + 34), fill=col, outline=(100, 100, 100), width=2)
            d.text(
                (x - 12, y - 18),
                ch,
                font=F(32),
                fill=(20, 20, 20) if side == 0 else (240, 240, 240),
            )
    return im


def receipt(store="FRESH MART", items=None, seed=0):
    items = items or [
        ("Milk 1L", 1.29),
        ("Bananas", 2.10),
        ("Bread", 2.49),
        ("Eggs x12", 3.99),
        ("Coffee beans", 8.90),
        ("Tomatoes", 2.75),
        ("Cheddar", 4.20),
    ]
    im = Image.new("RGB", (600, 1100), (230, 230, 225))
    d = ImageDraw.Draw(im)
    d.rectangle((60, 30, 540, 1070), fill=(253, 253, 248))
    d.text((180, 60), store, font=F(36), fill=(20, 20, 20))
    d.text((150, 115), "12 Market Street  Tel 555-0142", font=F(16), fill=(60, 60, 60))
    y = 190
    for name, p in items:
        d.text((90, y), name, font=F(22), fill=(30, 30, 30))
        d.text((420, y), f"{p:6.2f}", font=F(22), fill=(30, 30, 30))
        y += 50
    d.line((90, y + 10, 510, y + 10), fill=(60, 60, 60), width=2)
    d.text((90, y + 30), "TOTAL", font=F(28), fill=(10, 10, 10))
    d.text((400, y + 30), f"{sum(p for _, p in items):7.2f}", font=F(28), fill=(10, 10, 10))
    d.text((90, y + 90), "VISA ****4421   THANK YOU!", font=F(18), fill=(60, 60, 60))
    return im


def house(wall=(230, 200, 150), roof=(170, 60, 50), sky=(140, 190, 240)):
    im = Image.new("RGB", (1200, 800), sky)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 560, 1200, 800), fill=(80, 160, 70))
    d.rectangle((350, 330, 850, 620), fill=wall)
    d.polygon([(320, 340), (600, 150), (880, 340)], fill=roof)
    d.rectangle((560, 470, 650, 620), fill=(110, 70, 40))
    for x in (400, 720):
        d.rectangle((x, 380, x + 90, 460), fill=(200, 230, 250), outline=(255, 255, 255), width=5)
    d.rectangle((960, 420, 1000, 600), fill=(100, 70, 40))
    d.ellipse((880, 280, 1080, 480), fill=(50, 130, 50))
    for x in range(60, 300, 40):
        d.ellipse((x, 600, x + 20, 620), fill=(230, 80, 120))
    return im.filter(ImageFilter.GaussianBlur(1))


def train_jp():
    im, d = _win("時刻表 - 新宿駅 (中央線快速 東京方面)", jp=True)
    d.text((70, 100), "新宿駅 発車時刻", font=J(40), fill=(20, 20, 20))
    rows = [
        ("7時", "02 07 11 15 19 23 27 31 35 39 44 48 52 56"),
        ("8時", "00 04 08 12 16 20 24 28 32 36 40 45 50 55"),
        ("9時", "01 07 13 19 25 31 37 43 49 55"),
        ("10時", "02 10 18 26 34 42 50 58"),
    ]
    for i, (h, ms) in enumerate(rows):
        y = 190 + 90 * i
        d.rectangle((70, y, 1130, y + 76), outline=(180, 180, 180))
        d.text((90, y + 18), h, font=J(30), fill=(200, 80, 0))
        d.text((220, y + 22), ms, font=J(26), fill=(30, 30, 30))
    d.text((70, 580), "快速 = 黒  特別快速 = 赤  始発 = ◎", font=J(22), fill=(80, 80, 80))
    return im


def code_editor():
    im = Image.new("RGB", (1200, 760), (30, 32, 40))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1200, 40), fill=(45, 48, 58))
    d.text((20, 8), "app.js - Visual Studio Code", font=F(18), fill=(200, 200, 200))
    lines = [
        ("async function loadOrders(userId) {", (198, 120, 221)),
        ("  const res = await fetch(`/api/users/${userId}/orders`);", (220, 220, 220)),
        ("  if (!res.ok) throw new Error(`HTTP ${res.status}`);", (220, 220, 220)),
        ("  const data = await res.json();", (220, 220, 220)),
        ("  return data.orders.map(o => ({ id: o.id, total: o.total }));", (220, 220, 220)),
        ("}", (198, 120, 221)),
        ("", (0, 0, 0)),
        ("loadOrders(42).then(console.log).catch(console.error);", (97, 175, 239)),
    ]
    for i, (t, c) in enumerate(lines):
        d.text((20, 70 + 40 * i), f"{i + 1:>2}", font=F(20), fill=(100, 100, 110))
        d.text((70, 70 + 40 * i), t, font=F(22), fill=c)
    return im


def street_map():
    im = Image.new("RGB", (1200, 900), (240, 236, 226))
    d = ImageDraw.Draw(im)
    d.polygon(
        [
            (0, 600),
            (300, 520),
            (600, 560),
            (900, 470),
            (1200, 500),
            (1200, 580),
            (900, 550),
            (600, 640),
            (300, 600),
            (0, 680),
        ],
        fill=(140, 190, 235),
    )
    d.rectangle((700, 120, 1000, 380), fill=(170, 215, 150))
    d.text((760, 230), "City Park", font=F(26), fill=(30, 90, 30))
    for x in (150, 450, 650):
        d.line((x, 0, x, 900), fill="white", width=18)
    for y in (100, 420, 780):
        d.line((0, y, 1200, y), fill="white", width=18)
    d.line((450, 0, 450, 900), fill=(250, 210, 120), width=24)
    d.text((460, 300), "Main St", font=F(22), fill=(60, 60, 60))
    d.text((40, 640), "River Elbe", font=F(24), fill=(30, 70, 140))
    d.ellipse((140, 400, 170, 430), fill=(220, 50, 50))
    d.text((180, 395), "Station", font=F(22), fill=(60, 60, 60))
    return im


IMAGES = {
    "images/weather-app.png": weather_app,
    "images/tenki-tokyo.png": tenki_jp,
    "images/rainfall-chart.png": rainfall_chart,
    "images/stock-price.png": stock_chart,
    "images/height-weight-scatter.png": scatter,
    "images/org-chart.png": org_chart,
    "images/sequence-diagram.png": sequence_diagram,
    "images/music-player.png": music_player,
    "images/chess.png": chessboard,
    "images/receipt.jpg": receipt,
    "images/house.jpg": house,
    "images/train-timetable.png": train_jp,
    "images/code-editor.png": code_editor,
    "images/street-map.png": street_map,
}

# ------------------------------------------------------------------ mixed: queries
TEXT_Q = [
    ("how often to change the water in a fish tank", ["notes/aquarium.md"]),
    ("renew passport", ["notes/todo.txt"]),
    ("beer fermentation temperature", ["pdf/homebrew-beer.pdf"]),
    ("how much notice before moving out of the apartment", ["pdf/lease-agreement.pdf"]),
    ("protected bike lanes plan", ["pdf/cycling-strategy.pdf"]),
    ("speed limit in residential streets", ["pdf/cycling-strategy.pdf"]),
    ("resize a folder of photos from the terminal", ["projects/imgshrink/README.md"]),
    ("restic backup schedule", ["projects/backup.sh.md"]),
    ("compare insurance quotes", ["notes/car-insurance.md"]),
    ("weather forecast for the week", ["images/weather-app.png", "images/tenki-tokyo.png"]),
    ("which city gets the most rain", ["images/rainfall-chart.png"]),
    ("share price chart", ["images/stock-price.png"]),
    ("scatter plot of height and weight", ["images/height-weight-scatter.png"]),
    ("company organization chart", ["images/org-chart.png"]),
    ("sequence diagram of an API request to the database", ["images/sequence-diagram.png"]),
    ("music player showing the current song", ["images/music-player.png"]),
    ("chess board position", ["images/chess.png"]),
    ("grocery receipt total", ["images/receipt.jpg"]),
    ("a house with a garden", ["images/house.jpg"]),
    ("javascript code that fetches data from an api", ["images/code-editor.png"]),
    ("city map with a river and a park", ["images/street-map.png"]),
    ("train departure times", ["images/train-timetable.png"]),
    # Japanese
    ("味噌汁の作り方", ["notes/miso-soup.txt"]),
    ("オフィス移転の会議の議事録", ["notes/meeting-2026-09-12.md"]),
    ("家計簿のアプリ", ["projects/kakeibo/README.md"]),
    ("漢検の勉強方法", ["notes/kanji-study.md"]),
    ("オンライン英会話の進め方", ["notes/language-exchange.md"]),
    ("東京の天気予報", ["images/tenki-tokyo.png", "images/weather-app.png"]),
    ("新宿駅の電車の時刻表", ["images/train-timetable.png"]),
    # cross-lingual
    ("how to make miso soup", ["notes/miso-soup.txt"]),
    ("household budget web app", ["projects/kakeibo/README.md"]),
    ("買い物のレシート", ["images/receipt.jpg"]),
    ("チェスの盤面", ["images/chess.png"]),
]

NEG_Q = [
    "offside rule in ice hockey",
    "migration of monarch butterflies",
    "baroque harpsichord tuning",
    "symptoms of vitamin D deficiency",
    "how to become an airline pilot",
    "ancient egyptian burial rituals",
    "best surfing spots in portugal",
    "wine pairing for steak",
    "how to fix a leaking roof",
    "ブラックホールの蒸発",
    "相撲の番付の決め方",
    "京都の紅葉の名所",
    "猫の爪切りのコツ",
    "qwpoeiru zmxn",
    "foo bar baz",
    "xyz 123",
    "hello",
    "ああああ",
    "...",
]


def q_weather():
    return weather_app("Osaka", seed=9, accent=(220, 90, 40))


def q_chess():
    return chessboard(light=(235, 236, 208), dark=(119, 149, 86), seed=11)


def q_receipt():
    return receipt(
        "CORNER SHOP",
        [
            ("Apples", 3.10),
            ("Rice 2kg", 5.40),
            ("Yogurt", 1.80),
            ("Pasta", 1.25),
            ("Olive oil", 7.90),
        ],
    )


def q_line():
    return line_chart(
        "Average temperature in Oslo (C)",
        [-4, -4, 0, 5, 11, 15, 17, 16, 11, 6, 1, -3],
        (40, 120, 200),
        ["Jan", "Mar", "May", "Jul", "Sep", "Nov"],
    )


def q_org():
    return org_chart(
        {"Director": "J. Park", "Sales": "L. Chen", "Marketing": "R. Diaz", "Support": "T. Nowak"},
        color=(250, 230, 210),
    )


def q_house():
    return house(wall=(240, 240, 235), roof=(60, 70, 110), sky=(200, 210, 225))


def q_scatter_photo_negative():
    # Unrelated: a blurred abstract blob photo
    im = Image.new("RGB", (900, 700), (90, 60, 120))
    d = ImageDraw.Draw(im)
    rng = random.Random(4)
    for _ in range(25):
        x, y, r = rng.randint(0, 900), rng.randint(0, 700), rng.randint(30, 160)
        d.ellipse((x - r, y - r, x + r, y + r), fill=tuple(rng.randint(0, 255) for _ in range(3)))
    return im.filter(ImageFilter.GaussianBlur(25))


def q_hatching():
    im = Image.new("RGB", (800, 800), "white")
    d = ImageDraw.Draw(im)
    for k in range(-800, 800, 24):
        d.line((k, 0, k + 800, 800), fill=(40, 40, 40), width=6)
    return im


def q_radial():
    a = np.linspace(-1, 1, 600)
    r = np.sqrt(a[None] ** 2 + a[:, None] ** 2)
    img = (
        np.stack([255 * (1 - r / 1.5), 120 + 0 * r, 255 * r / 1.5], -1)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(img)


def q_paper():
    rng = np.random.default_rng(1)
    img = (225 + rng.normal(0, 6, (700, 900, 1))).clip(0, 255).repeat(3, -1).astype(np.uint8)
    return Image.fromarray(img).filter(ImageFilter.GaussianBlur(1.5))


IMAGE_Q = [
    ("weather-osaka", q_weather, ["images/weather-app.png", "images/tenki-tokyo.png"]),
    ("chess-green", q_chess, ["images/chess.png"]),
    ("receipt-other", q_receipt, ["images/receipt.jpg"]),
    ("line-temperature", q_line, ["images/stock-price.png"]),
    ("org-other", q_org, ["images/org-chart.png"]),
    ("house-blue", q_house, ["images/house.jpg"]),
]
NEG_IMAGE_Q = [
    ("blobs", q_scatter_photo_negative),
    ("hatching", q_hatching),
    ("radial", q_radial),
    ("paper", q_paper),
]

# ------------------------------------------------------------------ single-topic corpora
AKIYA_NOTES = {
    "akiya/akiya-bank.md": "# 空き家バンク\n\n市町村が空き家の売却・賃貸を希望する所有者の物件を登録し、移住希望者に紹介する仕組み。登録は無料。成約時の仲介は地元の宅建業者が行う。改修費の補助（上限50万円）と組み合わせると利用が増える。\n",
    "akiya/tokutei-akiya.md": "# 特定空家等の判定\n\n倒壊等の著しく保安上危険となるおそれ、著しく衛生上有害、景観を損なう、周辺の生活環境の保全に不適切、のいずれかに当たる空き家を特定空家等とする。判定は外観目視による調査票の点数で行い、助言・指導、勧告、命令、代執行の順に措置する。\n",
    "akiya/kaitai-hojo.md": "# 老朽空き家の解体費補助\n\n昭和56年以前に建てられた木造住宅で、不良度判定が100点以上のものが対象。解体工事費の3分の1、上限80万円を補助する。工事の契約前に申請が必要。\n",
    "akiya/kotei-shisanzei.md": "# 固定資産税の住宅用地特例\n\n住宅が建っている土地は固定資産税が最大6分の1に軽減される。特定空家等として勧告を受けると、翌年度からこの特例が外れ、税額が上がる。\n",
    "akiya/souzoku-touki.md": "# 相続登記の義務化\n\n2024年4月から、相続で不動産を取得したことを知った日から3年以内に相続登記をすることが義務になった。正当な理由なく怠ると10万円以下の過料。空き家の所有者不明化を防ぐ目的。\n",
    "akiya/kanri-daikou.md": "# 空き家の管理代行サービス\n\n遠方に住む所有者のかわりに、月1回の通気・通水、郵便物の整理、庭木の確認、外観の写真報告を行う。料金は月5,000円から1万円程度。\n",
}
AKIYA_Q = [
    ("空き家バンクに物件を登録する", ["akiya/akiya-bank.md"]),
    ("特定空家に指定される基準", ["akiya/tokutei-akiya.md"]),
    ("古い家の解体費用の補助金", ["akiya/kaitai-hojo.md"]),
    ("勧告を受けると固定資産税が上がる", ["akiya/kotei-shisanzei.md"]),
    ("相続登記はいつまでにするか", ["akiya/souzoku-touki.md"]),
    ("遠くに住んでいて空き家の管理ができない", ["akiya/kanri-daikou.md"]),
    ("inheritance registration deadline", ["akiya/souzoku-touki.md"]),
]
AKIYA_NEG = ["ラーメンのおいしい店", "how to tune a guitar", "asdf", "宇宙ステーションの生活"]

RECIPES = {
    "recipes/soft-boiled-eggs.md": "# Soft-boiled eggs\n\nLower eggs straight from the fridge into boiling water. Cook 6 minutes 30 seconds for a jammy yolk, then cool in ice water for 2 minutes before peeling.\n",
    "recipes/pancakes.md": "# Fluffy pancakes\n\n200 g flour, 2 tsp baking powder, 1 tbsp sugar, a pinch of salt, 1 egg, 300 ml milk, 30 g melted butter. Rest the batter 10 minutes. Cook on a medium pan until bubbles appear, then flip.\n",
    "recipes/guacamole.md": "# Guacamole\n\nMash 3 ripe avocados with lime juice and salt. Stir in finely chopped red onion, coriander and one jalapeno. Press cling film onto the surface so it does not turn brown.\n",
    "recipes/risotto.md": "# Mushroom risotto\n\nToast arborio rice in butter with shallots, deglaze with white wine, then add hot stock one ladle at a time, stirring, for about 18 minutes. Finish with parmesan and fried mushrooms.\n",
    "recipes/banana-bread.md": "# Banana bread\n\nUse 3 very ripe bananas, 75 g melted butter, 150 g sugar, 1 egg, 1 tsp baking soda, 190 g flour. Bake at 175 C for 55-60 minutes, until a skewer comes out clean.\n",
    "recipes/chili.md": "# Beef chili\n\nBrown 500 g minced beef with onion and garlic, add cumin, smoked paprika and chili powder, then tomatoes and kidney beans. Simmer for at least an hour; it tastes better the next day.\n",
    "recipes/caesar-salad.md": "# Caesar salad\n\nDressing: egg yolk, anchovies, garlic, lemon juice, Dijon mustard, olive oil and parmesan. Toss with romaine lettuce and garlic croutons.\n",
    "recipes/pad-thai.md": "# Pad thai\n\nSoak rice noodles in warm water for 20 minutes. Stir-fry shrimp and tofu, add the noodles with tamarind paste, fish sauce and palm sugar, then egg, bean sprouts and crushed peanuts.\n",
    "recipes/lemonade.md": "# Homemade lemonade\n\nMake a syrup of 150 g sugar in 150 ml hot water, cool it, add the juice of 6 lemons and 1 litre of cold water. Serve with ice and mint.\n",
    "recipes/roast-chicken.md": "# Roast chicken\n\nDry the skin, rub with butter, salt and thyme, put a halved lemon inside. Roast at 200 C for about 20 minutes per 500 g, rest 15 minutes before carving.\n",
}
RECIPE_Q = [
    ("how long to boil an egg for a runny yolk", ["recipes/soft-boiled-eggs.md"]),
    ("pancake batter", ["recipes/pancakes.md"]),
    ("keep avocado dip from going brown", ["recipes/guacamole.md"]),
    ("creamy rice dish with mushrooms", ["recipes/risotto.md"]),
    ("what to bake with overripe bananas", ["recipes/banana-bread.md"]),
    ("spicy beef and beans stew", ["recipes/chili.md"]),
    ("salad dressing with anchovies", ["recipes/caesar-salad.md"]),
    ("thai noodles with peanuts", ["recipes/pad-thai.md"]),
    ("refreshing summer drink", ["recipes/lemonade.md"]),
    ("oven temperature for a whole chicken", ["recipes/roast-chicken.md"]),
    ("卵のゆで時間", ["recipes/soft-boiled-eggs.md"]),
]
RECIPE_NEG = ["kubernetes pod eviction", "how to tune a guitar", "asdf"]

SOURDOUGH = {
    "notes/starter.md": "# Sourdough starter\n\nFeed the starter twice a day with equal weights of rye flour and water. It is ready when it doubles within 6 hours and smells mildly sour.\n",
    "notes/first-loaf.md": "# My first sourdough loaf\n\n100 g starter, 375 g water, 500 g bread flour, 10 g salt. The crumb was dense; next time a longer bulk fermentation.\n",
    "notes/bulk-fermentation.md": "# Bulk fermentation\n\nAt 24 C the dough needs about 5 hours. Do four sets of stretch and folds in the first two hours, then leave it until it has grown by half.\n",
    "notes/shaping.md": "# Shaping a boule\n\nPre-shape into a loose round, rest 20 minutes, then shape tightly and put it seam side up in a floured banneton.\n",
    "notes/cold-proof.md": "# Cold proof\n\nAn overnight proof in the fridge (12-16 hours at 4 C) gives a more sour flavour and makes scoring easier.\n",
    "notes/baking.md": "# Baking sourdough in a Dutch oven\n\nPreheat to 250 C with the pot inside. Bake 20 minutes with the lid on, then 25 minutes without it until the crust is deep brown.\n",
    "notes/scoring.md": "# Scoring patterns\n\nOne long cut at a shallow angle makes an ear. Use a razor blade on a lame and score quickly just before the loaf goes in the oven.\n",
    "notes/troubleshooting.md": "# Sourdough problems\n\nFlat loaf: over-proofed or weak starter. Big holes under the crust: under-proofed. Gummy crumb: cut before it cooled.\n",
    "notes/tax-deadline.md": "# Tax return\n\nThe deadline is 31 July. Collect the bank statements, the pension certificate and the receipts for the home office.\n",
    "notes/oil-change.md": "# Car service\n\nOil and filter change every 15,000 km. Last done at 48,200 km. Also check the brake pads next time.\n",
}
SOURDOUGH_Q = [
    (
        "sourdough bread",
        [
            f"notes/{n}.md"
            for n in (
                "starter",
                "first-loaf",
                "bulk-fermentation",
                "shaping",
                "cold-proof",
                "baking",
                "scoring",
                "troubleshooting",
            )
        ],
    ),
    ("how long should the dough rise", ["notes/bulk-fermentation.md", "notes/cold-proof.md"]),
    ("why is my loaf flat", ["notes/troubleshooting.md"]),
    ("when is the tax deadline", ["notes/tax-deadline.md"]),
]
SOURDOUGH_NEG = ["kubernetes pod eviction", "how to tune a guitar", "asdf", "hello"]


# ------------------------------------------------------------------ build
CORPORA = ("mixed", "recipes", "sourdough", "akiya")
QSETS = {
    "mixed": (TEXT_Q, NEG_Q, IMAGE_Q, NEG_IMAGE_Q),
    "recipes": (RECIPE_Q, RECIPE_NEG, [], []),
    "sourdough": (SOURDOUGH_Q, SOURDOUGH_NEG, [], []),
    "akiya": (AKIYA_Q, AKIYA_NEG, [], []),
}


def build(root: Path, name: str) -> None:
    """Write corpus `name` into the empty folder `root`."""
    texts = {"mixed": NOTES, "recipes": RECIPES, "sourdough": SOURDOUGH, "akiya": AKIYA_NOTES}[name]
    for rel, t in texts.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t, encoding="utf-8")
    if name != "mixed":
        return
    for rel, pages in SYN_PDFS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        demo.write_pdf(root / rel, pages)
    for rel, make in IMAGES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        im = make()
        im.save(p, quality=90) if p.suffix == ".jpg" else im.save(p)
