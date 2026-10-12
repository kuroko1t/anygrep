"""Build the reference vectors that the relevance cutoff compares a query against.

A query that matches nothing in particular ("asdf", "hello", "...") scores a
little higher against everything, so its best results can pass a fixed
minimum score. The median score of a fixed, varied set of documents and
pictures tells how high "everything" is for that query, independent of what
the user indexed (a folder of 50 sourdough notes must not make a sourdough
query look generic). This script embeds that set with the pinned model and
writes src/anygrep/reference.npz.

All texts and pictures are written or drawn here, so the file can be
redistributed with the package.

    python scripts/build_reference_set.py [--threads 2] [--out src/anygrep/reference.npz]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BUDGETS = (70, 140, 280)

# (title, text). Varied on purpose: prose, lists, code, logs, tables, mail,
# Japanese and a few other languages, on topics outside the test corpora.
TEXTS: list[tuple[str, str]] = [
    (
        "roman-roads",
        "The Romans built more than 80,000 km of paved roads. A typical road had "
        "layers of large stones, gravel and sand under a surface of fitted blocks, with a camber so that "
        "rain ran off into side ditches. Milestones gave the distance to the next town.",
    ),
    (
        "pond-ecology",
        "Small garden ponds attract frogs, newts and dragonflies within a season. "
        "Keep one side shallow so animals can climb out, avoid fish if you want amphibians, and let "
        "native plants such as water mint and marsh marigold spread along the edge.",
    ),
    (
        "index-funds",
        "An index fund buys every share in a market index in proportion to its size. "
        "Because nobody picks stocks, the yearly fee is low, often under 0.2 percent. Over long periods "
        "most actively managed funds do worse than the index after fees.",
    ),
    (
        "http-retry",
        "def fetch(url, attempts=3):\n    for i in range(attempts):\n        try:\n"
        "            return session.get(url, timeout=10)\n        except ConnectionError:\n"
        "            time.sleep(2 ** i)\n    raise RuntimeError(f'giving up on {url}')",
    ),
    (
        "nginx.conf",
        "server {\n  listen 443 ssl;\n  server_name example.org;\n  location /static/ "
        "{\n    root /var/www;\n    expires 30d;\n  }\n  location / {\n    proxy_pass "
        "http://127.0.0.1:8000;\n  }\n}",
    ),
    (
        "app.log",
        "2025-02-11 09:14:02 INFO worker-3 started job 8812\n2025-02-11 09:14:07 WARN "
        "worker-3 slow query (2.4 s) on orders\n2025-02-11 09:15:30 ERROR worker-1 lost connection to "
        "queue, reconnecting\n2025-02-11 09:15:31 INFO worker-1 reconnected",
    ),
    (
        "glaciers",
        "Glaciers move because ice deforms under its own weight and slides on meltwater "
        "at its base. Crevasses open where the ice speeds up or bends over a step in the bedrock. Many "
        "Alpine glaciers have lost more than half their volume since 1900.",
    ),
    (
        "chess-openings",
        "The Sicilian Defence answers 1.e4 with 1...c5 and leads to unbalanced "
        "positions. The Queen's Gambit offers a pawn on c4 to gain the centre. Beginners do best with "
        "simple development: knights before bishops, castle early, do not move the same piece twice.",
    ),
    (
        "meeting-2024-11-04",
        "Attendees: Sara, Tom, Priya. Decisions: move the release to the 18th; "
        "Tom owns the migration script. Open: who reviews the security checklist? Next meeting Monday "
        "10:00. Action items: Priya to send the draft to legal, Sara to book the room.",
    ),
    (
        "email-landlord",
        "Hello Mr Jensen, the kitchen tap has been dripping for two weeks and the "
        "bathroom fan stopped working. Could someone come by on Thursday or Friday afternoon? I am home "
        "after 3 pm. Thank you, Anna (flat 4B).",
    ),
    (
        "bird-migration",
        "Arctic terns fly from the Arctic to the Antarctic and back every year, about "
        "70,000 km. Birds navigate using the sun, the stars and the Earth's magnetic field, and young "
        "birds of many species find the route on their first trip without adults.",
    ),
    (
        "sql-report",
        "SELECT c.region, date_trunc('month', o.created_at) AS month,\n       "
        "sum(o.total) AS revenue, count(*) AS orders\nFROM orders o JOIN customers c ON c.id = "
        "o.customer_id\nWHERE o.status = 'paid'\nGROUP BY 1, 2\nORDER BY 1, 2;",
    ),
    (
        "photosynthesis",
        "Plants turn carbon dioxide and water into sugar using light. The light "
        "reactions in the thylakoid membranes split water and release oxygen; the Calvin cycle in the "
        "stroma fixes carbon. C4 and CAM plants reduce water loss in hot, dry places.",
    ),
    (
        "bike-commute",
        "Cycling 8 km to work takes me 25 minutes, about the same as the bus. In winter "
        "I use studded tyres and a front light of at least 400 lumen. A pannier keeps the back dry; a "
        "spare tube and a small pump live in it permanently.",
    ),
    (
        "contract-clause",
        "12. Termination. Either party may terminate this Agreement with ninety (90) "
        "days written notice. Sections 7 (Confidentiality) and 9 (Limitation of Liability) survive "
        "termination. Fees paid in advance for the period after termination will be refunded pro rata.",
    ),
    (
        "volcano-types",
        "Shield volcanoes such as Mauna Loa have gentle slopes built from runny "
        "basalt lava. Stratovolcanoes like Fuji alternate lava and ash layers and erupt explosively. "
        "Calderas form when the roof of an emptied magma chamber collapses.",
    ),
    (
        "piano-practice",
        "Practise slowly with a metronome, then raise the tempo by a few beats per "
        "minute once a passage is clean. Split hard bars into small groups and play them hands "
        "separately. Ten focused minutes a day beat one long session a week.",
    ),
    (
        "table-q3",
        "| Product | Q1 | Q2 | Q3 |\n|---|---|---|---|\n| Basic | 1,204 | 1,310 | "
        "1,288 |\n| Pro | 412 | 455 | 530 |\n| Team | 37 | 51 | 66 |\n| Total | 1,653 | 1,816 | 1,884 |",
    ),
    (
        "poem",
        "The kettle sings before the dawn,\nthe window holds a paler blue,\nthe street is quiet, "
        "the lamps are gone,\nand all the day is still to do.",
    ),
    (
        "dog-training",
        "Teach recall with a long line in a quiet field. Say the name once, reward "
        "generously when the dog arrives, and never call it to end the fun. Practise near distractions "
        "only after it works ten times out of ten at home.",
    ),
    (
        "css-grid",
        ".gallery {\n  display: grid;\n  grid-template-columns: repeat(auto-fill, "
        "minmax(180px, 1fr));\n  gap: 12px;\n}\n.gallery img {\n  width: 100%;\n  aspect-ratio: 4 / 3;\n"
        "  object-fit: cover;\n}",
    ),
    (
        "printing-press",
        "Gutenberg's press of around 1450 combined movable metal type, an oil-based "
        "ink and a screw press adapted from wine making. Within fifty years presses were working in "
        "more than 250 European towns and millions of books had been printed.",
    ),
    (
        "insurance-claim",
        "Claim number 55-2093. Water damage in the hallway after a pipe burst on 3 "
        "March. Attached: photos, the plumber's invoice and the repair quote. The adjuster visit is "
        "planned for 14 March between 9 and 12.",
    ),
    (
        "sleep",
        "Adults need seven to nine hours of sleep. A fixed wake-up time, daylight in the "
        "morning and a cool, dark bedroom help more than any supplement. Caffeine has a half-life of "
        "about five hours, so an afternoon coffee still affects sleep at night.",
    ),
    (
        "docker-compose.yml",
        "services:\n  db:\n    image: postgres:16\n    environment:\n      "
        "POSTGRES_PASSWORD: example\n    volumes:\n      - pgdata:/var/lib/postgresql/data\n  web:\n"
        '    build: .\n    ports:\n      - "8000:8000"\n    depends_on: [db]\nvolumes:\n  pgdata: {}',
    ),
    (
        "tides",
        "Tides come from the difference in the Moon's pull across the Earth. Spring tides, "
        "with the largest range, happen near full and new moon, when the Sun and Moon line up. The Bay "
        "of Fundy has the highest tides in the world, up to 16 metres.",
    ),
    (
        "job-ad",
        "We are hiring a part-time bookkeeper (20 hours a week) for a small architecture "
        "office. You will handle invoices, payroll and VAT returns. Experience with accounting "
        "software required. Send a CV and two references by the end of the month.",
    ),
    (
        "climbing",
        "Before each climb, check your partner: harness buckles doubled back, figure-eight "
        "knot finished with a stopper, belay device locked. Call out clearly, keep slack low near the "
        "ground and never lean back on a single piece of protection.",
    ),
    (
        "changelog",
        "## 2.4.0\n- Add CSV export for reports\n- Faster startup on large projects\n"
        "- Fix crash when a file name contains a newline\n## 2.3.1\n- Fix wrong time zone in "
        "reminders\n- Update translations (Polish, Korean)",
    ),
    (
        "ocean-plastic",
        "Most plastic in the ocean breaks into microplastics smaller than 5 mm, which "
        "fish and plankton swallow. Rivers carry a large share of it from land. Cleanup at sea is slow, "
        "so most effort goes into collecting waste before it reaches rivers.",
    ),
    (
        "knitting",
        "Cast on 40 stitches with 5 mm needles. Work in garter stitch for 4 rows, then "
        "stockinette until the piece measures 30 cm. Bind off loosely and block the scarf on a towel.",
    ),
    (
        "shell-history",
        "cd ~/src/site\ngit pull --rebase\nnpm ci\nnpm run build\nrsync -avz --delete "
        "dist/ deploy@host:/srv/site/\nssh deploy@host 'sudo systemctl reload nginx'",
    ),
    (
        "vaccines",
        "Vaccines train the immune system with a harmless piece or weakened form of a "
        "pathogen, so the body can respond fast to a real infection. Herd immunity protects people who "
        "cannot be vaccinated when enough of the population is immune.",
    ),
    (
        "wedding-plan",
        "Guest list 86 people. Venue deposit paid. Still to do: choose the menu "
        "(fish or vegetarian), book a photographer, order invitations by June, and find a band that "
        "plays after 11 pm. Budget left: 4,200.",
    ),
    (
        "algebra",
        "To solve a quadratic equation ax^2 + bx + c = 0, compute the discriminant "
        "d = b^2 - 4ac. If d > 0 there are two real roots, (-b +/- sqrt(d)) / 2a; if d = 0 one; if "
        "d < 0 the roots are complex conjugates.",
    ),
    (
        "japan-rail",
        "Shinkansen trains run every few minutes between Tokyo and Osaka at up to 285 "
        "km/h. Reserved seats can be booked a month ahead; unreserved cars are cheaper. Large luggage "
        "needs a seat with an oversized baggage area.",
    ),
    (
        "README-tool",
        "# fastcsv\n\nA small command line tool to filter and convert CSV files.\n\n"
        "## Install\n\n    cargo install fastcsv\n\n## Usage\n\n    fastcsv select name,age "
        "people.csv\n    fastcsv to-json people.csv > people.json",
    ),
    (
        "thermostat",
        "The thermostat keeps 20 C during the day and 17 C at night. Each degree lower "
        "saves roughly six percent of heating energy. Radiator valves in empty rooms can be set to the "
        "frost symbol.",
    ),
    (
        "invoice",
        "Invoice 2025-0147\nDate: 2 April 2025\nTo: Northwind Studio\n3 days of design "
        "work at 600 = 1,800\nTravel = 120\nSubtotal 1,920, VAT 20% 384, Total 2,304\nPayment within "
        "30 days to the account below.",
    ),
    (
        "aquarium-reef",
        "Coral reefs cover less than one percent of the sea floor but hold about a "
        "quarter of all marine species. Corals bleach when warm water makes them expel the algae that "
        "feed them; repeated heat waves can kill a reef.",
    ),
    (
        "unit-test",
        "def test_parse_date():\n    assert parse_date('2024-02-29') == date(2024, 2, 29)\n"
        "    with pytest.raises(ValueError):\n        parse_date('2023-02-29')\n    assert "
        "parse_date(' 2024-01-05 ') == date(2024, 1, 5)",
    ),
    (
        "moving-checklist",
        "- Cancel the internet contract (2 months notice)\n- Register the new "
        "address\n- Book the van for Saturday\n- Defrost the freezer\n- Take meter readings\n- Pack a box "
        "with tools, cables and kettle for the first day",
    ),
    (
        "ancient-greece",
        "Athenian democracy let adult male citizens vote directly in the assembly. "
        "Officials were often chosen by lot. Women, slaves and foreigners had no vote, so only a small "
        "part of the population took part.",
    ),
    (
        "translation-de",
        "Die Besprechung wurde auf Donnerstag verschoben. Bitte bringen Sie die "
        "aktuellen Zahlen aus dem Vertrieb mit und schicken Sie mir vorher eine kurze Zusammenfassung.",
    ),
    (
        "recette-fr",
        "Faire fondre le beurre, ajouter la farine et cuire une minute. Verser le lait "
        "petit a petit en remuant, saler, poivrer et ajouter un peu de muscade. Laisser epaissir a feu "
        "doux.",
    ),
    (
        "viaje-es",
        "El tren sale de Madrid a las 8:30 y llega a Sevilla a las 11:05. Hay que estar en "
        "el anden diez minutos antes. El billete de vuelta es flexible y se puede cambiar sin coste.",
    ),
    (
        "ja-gomi",
        "燃えるごみは月曜日と木曜日、朝8時までに集積所に出してください。資源ごみ（びん・缶・ペットボトル）"
        "は第2・第4水曜日です。粗大ごみは事前に電話で申し込みが必要です。",
    ),
    (
        "ja-kaigi",
        "議題：新製品の発売日について。決定事項：発売は10月15日とする。営業部は9月中に販売店向け"
        "資料を作成する。未決事項：価格は次回の会議で決める。次回は9月3日14時から。",
    ),
    (
        "ja-rekishi",
        "江戸時代には参勤交代によって大名が一年おきに江戸と領地を行き来した。その結果、街道や"
        "宿場町が整備され、各地の文化や物産が江戸に集まった。",
    ),
    (
        "ja-kenkou",
        "一日の歩数の目安は8000歩程度とされる。急に運動量を増やすと膝や足首を痛めやすいので、"
        "最初の数週間は少しずつ距離を延ばすとよい。水分補給も忘れずに。",
    ),
    (
        "ja-pc",
        "パソコンの動作が遅いときは、まず起動時に立ち上がるアプリを減らし、ディスクの空き容量を確認する。"
        "それでも改善しない場合はメモリの増設を検討する。",
    ),
    (
        "ja-tegami",
        "拝啓　秋も深まってまいりましたが、皆様いかがお過ごしでしょうか。先日はお心のこもった"
        "品をお送りいただき、誠にありがとうございました。家族一同で大切にいただいております。敬具",
    ),
    (
        "ja-tenki",
        "明日は西から天気が崩れ、昼過ぎから雨の降るところがある見込みです。最高気温は18度前後で、"
        "北風がやや強く吹くでしょう。傘を持ってお出かけください。",
    ),
    (
        "ja-keiyaku",
        "第5条（秘密保持）甲及び乙は、本契約に関して知り得た相手方の秘密情報を、相手方の書面による"
        "承諾なく第三者に開示してはならない。",
    ),
    (
        "ja-ryokou",
        "金沢の旅行メモ：兼六園は朝7時から開いているので早めに行く。昼は近江町市場で海鮮丼。"
        "夕方はひがし茶屋街を散歩して、金箔のソフトクリームを食べる。",
    ),
    (
        "ja-code",
        "# 設定ファイルを読み込む\nimport json\nwith open('config.json', encoding='utf-8') as f:\n"
        "    設定 = json.load(f)\nprint(設定['名前'])",
    ),
    ("zh-note", "下周一上午十点开会，讨论第三季度的预算。请各部门在周五之前把数据发给财务部。"),
    (
        "faq",
        "Q: Can I change my delivery address after ordering?\nA: Yes, until the parcel is "
        "dispatched. Open the order, choose Edit address and save.\nQ: How long do refunds take?\nA: "
        "Usually 3 to 5 working days after we receive the item.",
    ),
    (
        "astronomy",
        "A light-year is the distance light travels in a year, about 9.46 trillion km. "
        "The nearest star after the Sun, Proxima Centauri, is 4.2 light-years away. The Andromeda "
        "galaxy, 2.5 million light-years away, is visible to the naked eye.",
    ),
    (
        "woodwork",
        "Mark the cut with a knife, not a pencil, and saw on the waste side of the line. "
        "Clamp the board firmly, let the saw do the work and keep your eye above the blade. Sand with "
        "the grain, going from 120 to 240 grit.",
    ),
    (
        "support-ticket",
        "Ticket #4471: Customer cannot log in after resetting the password. The reset "
        "mail arrives, the new password is accepted, but the next login says 'invalid credentials'. "
        "Browser: Firefox 128. Steps tried: cleared cookies, other browser.",
    ),
    (
        "json-sample",
        '{\n  "id": 1042,\n  "name": "Desk lamp",\n  "price": 39.9,\n  "tags": '
        '["lighting", "office"],\n  "stock": {"warehouse": 18, "store": 3}\n}',
    ),
    (
        "economics",
        "Inflation measures how fast prices rise on average. Central banks raise interest "
        "rates to slow it: borrowing gets more expensive, spending falls and price pressure eases, "
        "usually with a delay of a year or more.",
    ),
    (
        "kids-story",
        "Once upon a time a small fox lived at the edge of a big forest. Every evening "
        "she counted the stars, and every evening she lost count at forty-two, because that was when "
        "the owl started telling jokes.",
    ),
]


def _load_demo():
    spec = importlib.util.spec_from_file_location(
        "make_demo_corpus", ROOT / "scripts" / "make_demo_corpus.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _images() -> list[tuple[str, Image.Image]]:
    """Varied synthetic pictures: photos, documents, UI, charts and plain textures."""
    demo = _load_demo()
    font = demo.font
    rng = np.random.default_rng(7)
    out: list[tuple[str, Image.Image]] = []

    im = Image.new("RGB", (1000, 700), (120, 170, 230))  # field under a sky
    d = ImageDraw.Draw(im)
    d.rectangle((0, 420, 1000, 700), fill=(90, 150, 60))
    d.ellipse((760, 80, 880, 200), fill=(250, 240, 180))
    for x in range(40, 1000, 140):
        d.rectangle((x + 20, 340, x + 34, 440), fill=(100, 70, 40))
        d.ellipse((x - 20, 260, x + 74, 360), fill=(40, 110, 50))
    out.append(("field", im))

    im = Image.new("RGB", (900, 1200), "white")  # page of text
    d = ImageDraw.Draw(im)
    for i, (_, t) in enumerate(TEXTS[:6]):
        words, line, y = t.split(), "", 80 + 150 * i
        for w in words:
            if len(line) + len(w) > 60:
                d.text((70, y), line, font=font(20), fill=(30, 30, 30))
                line, y = "", y + 26
            line += w + " "
        d.text((70, y), line, font=font(20), fill=(30, 30, 30))
    out.append(("document", im))

    im = Image.new("RGB", (1100, 700), (30, 32, 40))  # code editor
    d = ImageDraw.Draw(im)
    for i, line in enumerate(TEXTS[3][1].splitlines() + TEXTS[40][1].splitlines()):
        d.text((60, 40 + 34 * i), f"{i + 1:3d}  {line}", font=font(22), fill=(200, 210, 220))
    out.append(("code", im))

    im = Image.new("RGB", (1000, 650), "white")  # line chart
    d = ImageDraw.Draw(im)
    d.line((80, 560, 950, 560), fill="black", width=2)
    d.line((80, 60, 80, 560), fill="black", width=2)
    pts = [(80 + 70 * i, 500 - 30 * math.sin(i / 2) - 25 * i) for i in range(13)]
    d.line(pts, fill=(200, 60, 60), width=4)
    d.text((300, 15), "Visitors per month", font=font(28), fill="black")
    out.append(("line-chart", im))

    im = Image.new("RGB", (1000, 700), (235, 235, 240))  # web form
    d = ImageDraw.Draw(im)
    d.text((80, 50), "Create account", font=font(36), fill=(20, 20, 20))
    for i, lab in enumerate(["Name", "Email", "Phone", "Country"]):
        y = 140 + 110 * i
        d.text((80, y), lab, font=font(22), fill=(60, 60, 60))
        d.rectangle((80, y + 34, 700, y + 80), fill="white", outline=(160, 160, 160))
    d.rectangle((80, 600, 300, 650), fill=(30, 130, 80))
    d.text((130, 610), "Sign up", font=font(24), fill="white")
    out.append(("form", im))

    im = Image.new("RGB", (900, 900), (250, 250, 245))  # table
    d = ImageDraw.Draw(im)
    for r in range(12):
        for c in range(5):
            x, y = 60 + 160 * c, 60 + 66 * r
            d.rectangle((x, y, x + 160, y + 66), outline=(150, 150, 150))
            txt = ["Item", "Qty", "Price", "Tax", "Sum"][c] if r == 0 else f"{rng.integers(1, 999)}"
            d.text((x + 14, y + 20), txt, font=font(20), fill=(30, 30, 30))
    out.append(("table", im))

    im = Image.new("RGB", (800, 800), (240, 225, 200))  # portrait-like shapes
    d = ImageDraw.Draw(im)
    d.ellipse((250, 150, 550, 500), fill=(225, 180, 150))
    d.rectangle((200, 500, 600, 800), fill=(60, 80, 140))
    d.ellipse((330, 280, 360, 310), fill=(40, 40, 40))
    d.ellipse((440, 280, 470, 310), fill=(40, 40, 40))
    d.arc((340, 360, 460, 430), 20, 160, fill=(150, 60, 60), width=6)
    out.append(("person", im))

    im = Image.new("RGB", (1000, 800), (238, 236, 225))  # street map
    d = ImageDraw.Draw(im)
    for i in range(9):
        d.line((0, 90 * i + 20, 1000, 90 * i + 60), fill="white", width=14)
        d.line((120 * i + 30, 0, 120 * i - 20, 800), fill="white", width=10)
    d.polygon([(600, 300), (820, 260), (900, 520), (650, 560)], fill=(170, 210, 150))
    out.append(("map", im))

    im = Image.new("RGB", (1000, 700), (20, 20, 30))  # night city
    d = ImageDraw.Draw(im)
    for x in range(0, 1000, 70):
        h = int(rng.integers(150, 500))
        d.rectangle((x, 700 - h, x + 60, 700), fill=(45, 45, 60))
        for wy in range(700 - h + 15, 690, 30):
            for wx in range(x + 8, x + 55, 16):
                if rng.random() < 0.4:
                    d.rectangle((wx, wy, wx + 8, wy + 14), fill=(250, 220, 120))
    out.append(("night-city", im))

    im = Image.new("RGB", (900, 900), "white")  # icon grid
    d = ImageDraw.Draw(im)
    for i in range(16):
        x, y = 80 + 200 * (i % 4), 80 + 200 * (i // 4)
        col = tuple(int(v) for v in rng.integers(40, 220, 3))
        (d.ellipse if i % 2 else d.rounded_rectangle)((x, y, x + 140, y + 140), fill=col)
    out.append(("icons", im))

    grad = np.linspace(0, 1, 800)[None, :, None]  # soft gradient
    arr = np.array([40, 90, 160]) * (1 - grad) + np.array([240, 170, 120]) * grad
    out.append(("gradient", Image.fromarray(np.repeat(arr, 600, 0).astype(np.uint8))))

    noise = rng.normal(128, 40, (600, 800, 3)).clip(0, 255).astype(np.uint8)
    out.append(("texture", Image.fromarray(noise)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "src" / "anygrep" / "reference.npz")
    args = ap.parse_args()

    from anygrep.config import MODEL_REVISION, model_dir
    from anygrep.embedder import OnnxEmbedder

    emb = OnnxEmbedder(model_dir(), threads=args.threads)
    arrays: dict[str, np.ndarray] = {
        "text": emb.embed_documents([t for _, t in TEXTS], [n for n, _ in TEXTS]).astype(
            np.float16
        ),
        "model_revision": np.array(MODEL_REVISION),
    }
    images = _images()
    for b in BUDGETS:
        emb.image_budget = b
        arrays[f"image_{b}"] = np.stack([emb.embed_image(im) for _, im in images]).astype(
            np.float16
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"Wrote {args.out}: {len(TEXTS)} texts, {len(images)} images x {len(BUDGETS)} budgets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
