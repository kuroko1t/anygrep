#!/usr/bin/env bash
# Record demo/anygrep.gif with asciinema and agg (no browser needed).
#
#   scripts/record_demo.sh [WORKDIR]
#
# WORKDIR (default: a new temporary directory) receives the demo corpus and
# its index. It must be outside this repository, because the repository's
# .gitignore would hide the corpus from anygrep. The corpus is indexed before
# recording; the GIF shows only searches. The wait for each command's output
# (mostly loading the model) plays 6x faster; everything else is real time.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="${1:-$(mktemp -d)}"
mkdir -p "$work"
work="$(cd "$work" && pwd)"
cast="$work/anygrep.cast"

python "$repo/scripts/make_demo_corpus.py" "$work/demo-corpus"
anygrep model --download # keep the download out of the recording
export ANYGREP_DB="$work/demo-index.sqlite"
rm -f "$ANYGREP_DB" "$ANYGREP_DB-wal" "$ANYGREP_DB-shm" "$ANYGREP_DB-lock"
cd "$work/demo-corpus"
anygrep index -q . # searches run here (PATH defaults to the current folder)

asciinema rec --overwrite -q --cols 92 --rows 16 \
    -c "bash $repo/scripts/demo_session.sh" "$cast"
python "$repo/scripts/speed_up_cast.py" "$cast" "$cast.fast" \
    --from $'\033[22m' --to $'\033[23m' --factor 6 --every
agg --font-family "DejaVu Sans Mono" --font-size 20 --line-height 1.3 \
    --theme monokai --idle-time-limit 3 --last-frame-duration 3 \
    "$cast.fast" "$repo/demo/anygrep.gif"
echo "Wrote $repo/demo/anygrep.gif"
