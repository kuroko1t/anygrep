"""Play one part of an asciicast v2 recording faster.

    python scripts/speed_up_cast.py IN.cast OUT.cast --from TEXT --to TEXT [--factor 4]

Events between the first output containing --from and the next output
containing --to are played --factor times faster; the output itself is
unchanged. With --every, every such part is sped up, not only the first.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--factor", type=float, default=4.0)
    ap.add_argument("--every", action="store_true", help="speed up every matching part")
    a = ap.parse_args()
    with open(a.src, encoding="utf-8") as f:
        header, *events = f.read().splitlines()
    out = [header]
    state, prev_in, prev_out = "before", 0.0, 0.0
    text = ""
    for line in events:
        t, kind, data = json.loads(line)
        dt = t - prev_in
        prev_in = t
        if state == "inside":
            dt /= a.factor
        prev_out += dt
        out.append(json.dumps([round(prev_out, 6), kind, data], ensure_ascii=False))
        text = (text + data)[-200:]
        if state == "before" and a.start in text:
            state, text = "inside", ""
        elif state == "inside" and a.end in text:
            state, text = ("before" if a.every else "after"), ""
    with open(a.dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
