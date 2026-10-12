"""Generate a small, reproducible demo corpus: Markdown notes, PDFs, images.

All content is synthetic and written for this project (no third-party text
or pictures), so the corpus can be shared freely.

    python scripts/make_demo_corpus.py [OUTPUT_DIR]   # default: demo/corpus
"""

from __future__ import annotations

import math
import random
import sys
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

NOTES: dict[str, str] = {
    "notes/kyoto-trip.md": """# Kyoto trip, November

## Day 1
Arrive at Kyoto station around noon, drop bags at the hotel near Shijo.
Afternoon walk through Gion and up to Kiyomizu-dera before sunset.

## Day 2
Early start at Fushimi Inari to avoid the crowds; the full loop up the
mountain takes about two and a half hours. Lunch: kitsune udon.
Evening: Tofuku-ji for the autumn leaves, peak colour is usually late November.

## Day 3
Arashiyama bamboo grove at 7am, then the monkey park. Rent bicycles and ride
along the river. Buy matcha sweets for the office.

## Budget
Hotel 3 nights 54,000 yen, rail pass not needed, around 8,000 yen per day for
food and temple entrance fees.
""",
    "notes/sourdough.md": """# Sourdough bread, the version that finally worked

Starter: feed 1:1:1 the night before, it should double in 6-8 hours.

1. Mix 450 g bread flour, 50 g whole wheat, 350 g water. Rest 1 hour (autolyse).
2. Add 100 g active starter and 10 g salt, squeeze until combined.
3. Bulk fermentation: 4 to 5 hours at 24 C, with four sets of stretch and folds
   in the first two hours. The dough is ready when it has grown by about half
   and is jiggly with bubbles on the sides.
4. Pre-shape, bench rest 20 minutes, final shape into a tight boule.
5. Cold proof in the fridge overnight, 12 to 16 hours.
6. Bake in a preheated Dutch oven at 250 C: 20 minutes with the lid, then
   25 minutes at 230 C without the lid, until deep brown.

Let it cool for at least an hour before cutting, otherwise the crumb is gummy.
""",
    "notes/meetings/2026-03-budget.md": """# Budget meeting 2026-03-12

Attendees: Mia, Jonas, Priya, me

## Decisions
- Marketing budget for Q2 is cut by 15 percent; the conference sponsorship is
  dropped, the newsletter stays.
- Hiring: one backend engineer approved, the designer position moves to Q3.
- Cloud costs: Jonas will move the batch jobs to reserved instances, expected
  saving about 2,000 per month.

## Action items
- Priya: send the revised forecast to finance by Friday.
- Me: write the job description for the backend role.
- Jonas: report the reserved instance savings at the next meeting.
""",
    "notes/dev/rust-borrow-checker.md": """# Rust: borrow checker notes

A value can have either one mutable reference or any number of shared
references at a time, never both. Most "cannot borrow as mutable because it is
also borrowed as immutable" errors come from holding a reference into a
vector while pushing to it.

Fixes that usually work:
- Clone the small value instead of keeping the reference.
- Limit the scope of the borrow with an inner block.
- Store indices instead of references.
- Use `std::mem::take` to move a field out temporarily.

Lifetimes annotate how long references are valid; `'static` means the data
lives for the whole program, for example string literals.
""",
    "notes/dev/docker-cheatsheet.md": """# Docker cheat sheet

- List running containers: `docker ps`, all containers: `docker ps -a`
- Remove dangling images: `docker image prune`
- Free disk space used by build cache: `docker builder prune --all`
- Follow logs: `docker logs -f <container>`
- Open a shell inside a container: `docker exec -it <container> sh`
- Compose up in the background: `docker compose up -d`

If the disk fills up, check `docker system df` first; old build cache is
usually the culprit.
""",
    "notes/home/houseplants.md": """# Houseplants

Succulents and cacti: water only when the soil is completely dry, roughly every
two to three weeks in summer and once a month in winter. Bright light, south
window. Overwatering is what kills them, look for soft, translucent leaves.

Monstera: water weekly, likes humidity, wipe the leaves. Yellow leaves usually
mean too much water.

Basil on the balcony: pinch off the flower buds so it keeps producing leaves.
""",
    "notes/health/half-marathon-plan.md": """# Half marathon training plan (12 weeks)

Goal: finish under 2 hours, race on June 7.

- Three runs per week: one easy run, one interval or tempo session, one long run.
- Long run grows from 8 km to 18 km, cut back every fourth week.
- Intervals: 6 x 800 m at 5:10 per km with 2 minutes of jogging between.
- Strength training twice a week: squats, lunges, calf raises, planks.
- Taper: the last 10 days drop volume by 40 percent, keep a bit of speed.

Shoes: replace after about 700 km. Knee pain last spring came from increasing
the weekly distance too fast.
""",
    "notes/finance/tax-checklist.md": """# Tax return checklist

- Salary statement from the employer
- Receipts for medical expenses above the threshold
- Donations to registered charities
- Home office: square metres, share of rent and electricity
- Interest and dividends statements from the bank and the broker
- Invoices for the side project (freelance income) and related expenses

Deadline reminder: file before the end of the month to avoid the late fee.
""",
    "notes/finance/kakutei-shinkoku.md": """# 確定申告メモ

- 期間は毎年2月16日から3月15日まで。e-Taxならスマホとマイナンバーカードで提出できる。
- 医療費控除: 1年間の医療費が10万円を超えた分が対象。領収書は5年間保管する。
- ふるさと納税をした場合は寄附金受領証明書を用意する。
- 副業の所得が20万円を超えたら申告が必要。
""",
    "notes/reading/habits.md": """# Notes on building habits

Small habits compound. Make the cue obvious (put the running shoes by the door),
make the action easy (two-minute version first), and make it satisfying
(track it on a calendar). Stacking a new habit on an existing one works well:
"after I pour my morning coffee, I write three lines in the journal".

Breaking a bad habit is the inverse: make it invisible and inconvenient.
Phone in another room during deep work.
""",
    "notes/ideas.txt": """Ideas, unsorted

- weekend project: a small weather station on the balcony with a Raspberry Pi
  and a temperature and humidity sensor, push readings to a local dashboard
- write a blog post about the sourdough experiments
- learn to sail this summer, beginner course at the lake
- birthday present for Lena: a film camera and a few rolls of film
""",
    "notes/recipes/vegetable-curry.md": """# Weeknight vegetable curry

Onion, garlic and ginger in oil until soft. Add 2 tbsp curry paste, then sweet
potato, chickpeas, a can of coconut milk and a can of tomatoes. Simmer 25
minutes, add spinach at the end, lime juice and coriander. Serve with rice.
Freezes well, makes four portions.
""",
}

PDFS: dict[str, list[list[str]]] = {
    "docs/home-network-manual.pdf": [
        [
            "Home Network Setup Guide",
            "",
            "1. Connect the router to the fibre modem with the blue cable.",
            "2. Power on and wait until the status light turns solid white.",
            "3. Open http://192.168.1.1 in a browser to reach the admin page.",
            "The default admin password is printed on the label under the router.",
        ],
        [
            "Changing the Wi-Fi name and password",
            "",
            "Log in to the admin page and open Wireless > Security.",
            "Choose WPA3, enter a new passphrase of at least 12 characters,",
            "and press Save. All devices must reconnect with the new password.",
            "If you forgot the admin password, hold the reset button on the back",
            "for 10 seconds to restore factory settings.",
        ],
        [
            "Troubleshooting slow connections",
            "",
            "Move the router away from the microwave and metal shelves.",
            "Use the 5 GHz band for laptops close to the router, 2.4 GHz for range.",
            "Restart the router once a month and keep the firmware updated.",
        ],
    ],
    "docs/employee-handbook.pdf": [
        [
            "Employee Handbook - Example Company",
            "",
            "Welcome to the team. This handbook explains how we work together,",
            "our working hours, remote work rules and the benefits we offer.",
        ],
        [
            "Working hours and remote work",
            "",
            "Core hours are 10:00 to 15:00. Employees may work remotely up to",
            "three days per week after the probation period of three months.",
        ],
        [
            "Vacation and leave",
            "",
            "New employees receive 25 paid vacation days per calendar year,",
            "prorated in the first year. Unused days can be carried over until",
            "the end of March of the following year. Sick leave requires a",
            "doctor's note from the third day.",
        ],
    ],
    "docs/solar-panel-report.pdf": [
        [
            "Rooftop Solar - First Year Report",
            "",
            "The 6.2 kWp system produced 6,850 kWh in its first year.",
            "Production peaked in June with 870 kWh and was lowest in December",
            "with 160 kWh. Self-consumption was 38 percent; the rest was exported",
            "to the grid. Estimated payback period: 9 years.",
        ],
        [
            "Maintenance",
            "",
            "Clean the panels once a year after the pollen season.",
            "Check the inverter log monthly for error codes.",
        ],
    ],
}


# ---------------------------------------------------------------- tiny PDF writer
def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a text-only PDF (Helvetica, A4) with one list of lines per page."""

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objs: list[bytes] = []
    n_pages = len(pages)
    # 1: catalog, 2: pages, 3: font, then (page, content) pairs
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        ops = ["BT", "/F1 16 Tf", "72 770 Td", "20 TL"]
        for j, line in enumerate(lines):
            if j == 1:
                ops.append("/F1 11 Tf")
            ops.append(f"({esc(line)}) Tj T*")
        ops.append("ET")
        stream = zlib.compress("\n".join(ops).encode("latin-1"))
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode()
        )
        objs.append(
            f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------- images
def font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def error_dialog() -> Image.Image:
    im = Image.new("RGB", (1280, 800), (32, 96, 160))
    d = ImageDraw.Draw(im)
    d.rectangle((340, 250, 940, 550), fill=(240, 240, 240), outline=(90, 90, 90), width=2)
    d.rectangle((340, 250, 940, 290), fill=(200, 60, 60))
    d.text((355, 258), "Error", font=font(22), fill="white")
    d.ellipse((370, 330, 430, 390), fill=(200, 40, 40))
    d.text((392, 336), "!", font=font(40), fill="white")
    d.text((460, 330), "Disk full", font=font(30), fill=(20, 20, 20))
    d.text((460, 375), "Not enough free space to save", font=font(20), fill=(40, 40, 40))
    d.text((460, 402), "'report-final.docx'. Free up space", font=font(20), fill=(40, 40, 40))
    d.text((460, 429), "on drive C: and try again.", font=font(20), fill=(40, 40, 40))
    d.rectangle((800, 490, 910, 530), fill=(225, 225, 225), outline=(120, 120, 120))
    d.text((838, 498), "OK", font=font(20), fill=(20, 20, 20))
    return im


def terminal_traceback() -> Image.Image:
    im = Image.new("RGB", (1200, 700), (24, 24, 28))
    d = ImageDraw.Draw(im)
    lines = [
        ("$ python train.py --epochs 10", (180, 220, 180)),
        ("Traceback (most recent call last):", (230, 230, 230)),
        ('  File "train.py", line 3, in <module>', (230, 230, 230)),
        ("    import numpy as np", (230, 230, 230)),
        ("ModuleNotFoundError: No module named 'numpy'", (255, 110, 110)),
        ("$ pip install numpy", (180, 220, 180)),
    ]
    d.rectangle((0, 0, 1200, 36), fill=(60, 60, 66))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((14 + 26 * i, 10, 30 + 26 * i, 26), fill=c)
    for i, (t, c) in enumerate(lines):
        d.text((24, 60 + 42 * i), t, font=font(24), fill=c)
    return im


def bar_chart(title: str, values: list[float], labels: list[str], color) -> Image.Image:
    im = Image.new("RGB", (1000, 650), "white")
    d = ImageDraw.Draw(im)
    d.text((60, 25), title, font=font(30), fill=(20, 20, 20))
    x0, y0, x1, y1 = 90, 100, 960, 560
    d.line((x0, y1, x1, y1), fill="black", width=2)
    d.line((x0, y0, x0, y1), fill="black", width=2)
    vmax = max(values)
    bw = (x1 - x0) / len(values)
    for i, (v, lab) in enumerate(zip(values, labels, strict=True)):
        h = (y1 - y0 - 20) * v / vmax
        bx = x0 + i * bw + bw * 0.15
        d.rectangle((bx, y1 - h, bx + bw * 0.7, y1), fill=color)
        d.text((bx, y1 + 10), lab, font=font(16), fill=(40, 40, 40))
    return im


def flowchart() -> Image.Image:
    im = Image.new("RGB", (1100, 760), (250, 250, 250))
    d = ImageDraw.Draw(im)
    f = font(22)
    boxes = [
        ((420, 40, 680, 110), "User opens app"),
        ((420, 170, 680, 240), "Enter email + password"),
        ((420, 300, 680, 370), "Check credentials"),
        ((140, 450, 400, 520), "Show error message"),
        ((700, 450, 960, 520), "Send 2FA code"),
        ((700, 600, 960, 670), "Login successful"),
    ]
    for b, t in boxes:
        d.rounded_rectangle(b, radius=14, fill=(220, 235, 255), outline=(40, 80, 160), width=3)
        d.text((b[0] + 16, b[1] + 22), t, font=f, fill=(10, 30, 70))
    arrows = [((550, 110), (550, 170)), ((550, 240), (550, 300)), ((480, 370), (300, 450)),
              ((620, 370), (800, 450)), ((830, 520), (830, 600))]  # fmt: skip
    for a, b in arrows:
        d.line((a, b), fill=(40, 40, 40), width=3)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        p1 = (b[0] - 14 * math.cos(ang - 0.4), b[1] - 14 * math.sin(ang - 0.4))
        p2 = (b[0] - 14 * math.cos(ang + 0.4), b[1] - 14 * math.sin(ang + 0.4))
        d.polygon([b, p1, p2], fill=(40, 40, 40))
    d.text((300, 390), "invalid", font=font(18), fill=(160, 40, 40))
    d.text((720, 390), "valid", font=font(18), fill=(40, 120, 40))
    return im


def sunset() -> Image.Image:
    w, h = 1280, 853
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    horizon = int(h * 0.58)
    for y in range(horizon):
        t = y / horizon
        d.line((0, y, w, y), fill=(int(70 + 180 * t), int(40 + 90 * t), int(120 - 60 * t)))
    d.ellipse((560, horizon - 110, 720, horizon + 50), fill=(255, 190, 80))
    rng = random.Random(3)
    for y in range(horizon, h):
        t = (y - horizon) / (h - horizon)
        d.line(
            (0, y, w, y), fill=(int(40 + 60 * (1 - t)), int(40 + 40 * (1 - t)), int(90 + 30 * t))
        )
    for _ in range(140):
        y = rng.randint(horizon + 5, h - 5)
        x = rng.randint(0, w)
        ln = rng.randint(20, 120)
        d.line(
            (x, y, x + ln, y),
            fill=(255, 170, 90) if abs(x - 640) < 180 else (120, 120, 160),
            width=2,
        )
    d.rectangle((0, h - 90, w, h), fill=(210, 190, 150))  # sand
    return im


def bicycle() -> Image.Image:
    im = Image.new("RGB", (1000, 700), (235, 240, 245))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 560, 1000, 700), fill=(150, 150, 150))
    red = (200, 30, 40)
    for cx in (290, 710):
        d.ellipse((cx - 150, 410 - 150, cx + 150, 410 + 150), outline=(20, 20, 20), width=14)
        for k in range(12):
            a = k * math.pi / 6
            d.line(
                (cx, 410, cx + 140 * math.cos(a), 410 + 140 * math.sin(a)),
                fill=(120, 120, 120),
                width=2,
            )
    pts = {
        "rear": (290, 410),
        "crank": (480, 420),
        "seat": (430, 230),
        "head": (660, 220),
        "front": (710, 410),
    }
    for a, b in [("rear", "crank"), ("crank", "seat"), ("seat", "rear"), ("seat", "head"),
                 ("head", "crank"), ("head", "front")]:  # fmt: skip
        d.line((pts[a], pts[b]), fill=red, width=16)
    d.rectangle((395, 205, 480, 225), fill=(30, 30, 30))  # saddle
    d.line((640, 180, 720, 170), fill=(30, 30, 30), width=12)  # handlebar
    d.line((660, 220, 650, 180), fill=red, width=12)
    return im


def whiteboard() -> Image.Image:
    im = Image.new("RGB", (1200, 800), (252, 252, 248))
    d = ImageDraw.Draw(im)
    d.rectangle((20, 20, 1180, 780), outline=(160, 160, 160), width=8)
    d.text((80, 60), "Q3 ROADMAP", font=font(54), fill=(20, 60, 160))
    items = ["1. Launch mobile app beta", "2. Migrate billing to new provider",
             "3. Improve search speed", "4. Hire 2 support engineers"]  # fmt: skip
    for i, t in enumerate(items):
        d.text((100, 190 + 90 * i), t, font=font(38), fill=(30, 30, 30))
    d.ellipse((880, 520, 1100, 720), outline=(200, 40, 40), width=6)
    d.text((925, 595), "July!", font=font(40), fill=(200, 40, 40))
    return im


def pie_chart() -> Image.Image:
    im = Image.new("RGB", (900, 650), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 20), "Household spending 2025", font=font(30), fill=(20, 20, 20))
    parts = [("Rent", 42, (70, 110, 200)), ("Food", 21, (240, 150, 40)),
             ("Transport", 12, (90, 170, 90)), ("Travel", 10, (200, 80, 160)),
             ("Other", 15, (150, 150, 150))]  # fmt: skip
    start = -90.0
    for _name, pct, col in parts:
        end = start + 360 * pct / 100
        d.pieslice((80, 100, 580, 600), start, end, fill=col, outline="white", width=3)
        start = end
    for i, (name, pct, col) in enumerate(parts):
        d.rectangle((640, 180 + 50 * i, 670, 210 + 50 * i), fill=col)
        d.text((685, 182 + 50 * i), f"{name} {pct}%", font=font(22), fill=(30, 30, 30))
    return im


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo/corpus")
    for rel, text in NOTES.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    for rel, pages in PDFS.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        write_pdf(p, pages)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    images = {
        "screenshots/error-dialog.png": error_dialog(),
        "screenshots/terminal.png": terminal_traceback(),
        "screenshots/electricity-usage.png": bar_chart(
            "Monthly electricity usage (kWh)",
            [420, 390, 350, 300, 260, 240, 250, 255, 280, 330, 380, 430],
            months,
            (60, 120, 200),
        ),
        "diagrams/login-flow.png": flowchart(),
        "diagrams/spending-pie.png": pie_chart(),
        "photos/sunset-beach.jpg": sunset(),
        "photos/red-bicycle.jpg": bicycle(),
        "photos/whiteboard.jpg": whiteboard(),
    }
    for rel, im in images.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        im.save(p, quality=90) if p.suffix == ".jpg" else im.save(p)
    print(f"Wrote {len(NOTES)} notes, {len(PDFS)} PDFs, {len(images)} images to {out}")


if __name__ == "__main__":
    main()
