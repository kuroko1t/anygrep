#!/usr/bin/env bash
# The terminal session shown in demo/anygrep.gif. Run by scripts/record_demo.sh
# inside `asciinema rec` in the demo corpus folder after it has been indexed
# (so no PATH is needed and no indexing shows); every command really runs, only
# the typing is simulated.
set -uo pipefail

# Invisible escape sequences around each command's run, so that record_demo.sh
# can shorten the wait for the model to load. The output itself is unchanged.
START=$'\033[22m'
END=$'\033[23m'

type_cmd() {
    printf '\033[1;32m$\033[0m '
    local s="$1" i
    for ((i = 0; i < ${#s}; i++)); do
        printf '%s' "${s:i:1}"
        sleep 0.015
    done
    sleep 0.3
    printf '\n'
}

# run COMMAND [COMMENT] [PAUSE]: COMMENT is typed as a shell comment, not run.
# PAUSE is how long the result stays on screen before the next command.
run() {
    type_cmd "$1${2:+   # $2}"
    printf '%s' "$START"
    eval "$1"
    printf '%s' "$END"
    sleep "${3:-3.5}"
}

clear
run "anygrep 'running out of disk space'"
echo
run "anygrep 'パン生地の発酵時間'" "Japanese query, English note"
echo
run "anygrep 'how many vacation days do I get'" "" 0.5
printf '\033[1;32m$\033[0m '
sleep 0.5
