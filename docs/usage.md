# Usage

Everything here is also summarised by `anygrep -h` and `anygrep <command> -h`.

## Installing

```sh
pipx install anygrep                                     # or: uv tool install anygrep
pipx install git+https://github.com/kuroko1t/anygrep     # latest commit instead of the PyPI release
```

Python 3.10 or newer on Linux, Windows or macOS 13 or newer. No PyTorch, no CUDA. On first use anygrep
downloads the int8 model files from Hugging Face, about 540 MB (346 MB for text, 195 MB for images;
`anygrep model` lists them), into your user cache directory.

onnxruntime has no Intel Mac wheels for Python 3.14 and no Windows-on-ARM wheels for Python 3.10, so
the install fails there. On an Intel Mac use Python 3.13 or older
(`pipx install --python python3.13 anygrep`); pip then installs onnxruntime 1.23. On Windows on ARM
use Python 3.11 or newer.

To use anygrep from your own Python code, install it with pip into that project's environment instead;
see [python-api.md](python-api.md).

### Uninstalling

Run `pipx uninstall anygrep` (or `uv tool uninstall anygrep`), then delete the index and model files (`anygrep status` shows both
paths). They live in `~/.local/share/anygrep` and `~/.cache/anygrep` on Linux,
`~/Library/Application Support/anygrep` and `~/Library/Caches/anygrep` on macOS, and
`%LOCALAPPDATA%\anygrep` on Windows. `anygrep reset` deletes only the index.

## Example

```console
$ cd demo-corpus             # notes, PDFs, screenshots; see "Demo corpus" below to generate it
$ anygrep 'running out of disk space'
Indexing 23/23 100%|████████████████████████████████████| [00:13<00:00]
anygrep: added ~/anygrep-demo/demo-corpus as an indexed folder (23 files). Undo: anygrep rm ~/anygrep-demo/demo-corpus
 1  0.767  screenshots/error-dialog.png  [image]
$ anygrep 'how many vacation days do I get'
 1  0.762  docs/employee-handbook.pdf  p.3  [pdf]
      Vacation and leave New employees receive 25 paid vacation days per calendar year...
$ anygrep 'パン生地の発酵時間'   # query in Japanese, English note
 1  0.713  notes/sourdough.md:1
      # Sourdough bread, the version that finally worked Starter: feed 1:1:1 the night be...
```

The first search indexed the folder with a progress bar (13 seconds with 8 threads, see
[benchmarks.md](benchmarks.md)); later ones only index what changed.
`error-dialog.png` is a screenshot of a "Disk full" dialog, and `パン生地の発酵時間` is Japanese for "dough
proofing time": queries and documents can be in different languages. Weak matches are hidden, so a query
can return fewer than `-m` results.

## Indexing

A search indexes new and changed files under the folder it searches first: on a terminal all of them,
with a progress bar, and in scripts only when that is quick (see "Indexing before a search" below).
`anygrep index [PATH ...]` does it ahead of time, for example from cron, and `anygrep index --background
PATH` in a low-priority process. It is incremental. Without paths it updates every folder and file
indexed before.

- `--image-tokens 70|140|280` sets how many vision tokens each image gets. The default is 140. Use 70 for
  a faster first pass over many images, and 280 for screenshots or photos of documents where small text
  matters. Embeddings depend on the budget, so changing it re-embeds the images already indexed. The
  setting is stored in the index; if the change is interrupted, a plain `anygrep index` finishes it.
  Searches use each image's stored budget for relevance filtering. If an image query searches images
  with different budgets, it is embedded at each required budget, which adds some query time.
- `--no-images` skips images for this run and keeps images that are already indexed.
- `--ext .py --ext .json` also indexes those extensions as plain text. These are remembered.
- `--max-size MB` takes a finite positive number and skips files larger than this in this run.
  Larger files that are already indexed are kept.
- `-q` hides progress output.
- `--dry-run` shows how many files would be indexed or removed, and changes nothing:
  `~/Pictures: 2,314 new or changed files (412 text, 0 PDF, 1,902 images), 37 to remove`.
- `--background PATH ...` indexes in a detached, low-priority process (fewer threads, lower priority),
  so you can keep working and searching; see "Large folders" below. `--stop` stops it.

Common options for every command:

- `-j N`, `--threads N`: CPU threads for the model. Default: half the logical CPUs, at most 8. Also
  `ANYGREP_THREADS`. On an idle machine with more cores, `-j 12` can be faster (see
  [benchmarks.md](benchmarks.md#threads)).
- `--db FILE`: the index file. Default: a SQLite file in your user data directory
  (`anygrep status` shows the path). Also `ANYGREP_DB`.

Indexing can be interrupted at any time (Ctrl-C) and continues where it stopped. Text and PDFs are
indexed before images, and images newest first, so an interrupted run has made the cheap and most
searched files findable. Only one process changes the index at a time: an `index`, `remove` or `reset`
started while another one runs exits with status 2, and a search searches the index as it is.

### Large folders

`anygrep index --background ~/Pictures` starts one background job per index and returns at once:

- The job runs at a low priority with half the threads (`-j N` sets them), commits every few seconds,
  and keeps running when the terminal is closed.
- `anygrep status` shows it with the same progress line as the progress bar
  (`Indexing 340/1363  25%|##5       | [12:03<35:10]`). The remaining time is the job's own
  measurement and shows `?` until there is one.
- Searches meanwhile search what is indexed so far and end with that progress line.
- `anygrep index --background OTHER` while a job runs adds `OTHER` to it (it is indexed next).
- `anygrep index --stop` stops it; the next `anygrep index --background PATH` (or `anygrep index`)
  continues. A job that does not stop within 30 seconds is killed, but only when its process is
  still the job's (same PID and start time); otherwise `--stop` fails with status 2. While a job
  runs, `index`, `remove` and `reset` exit with status 2.
- A job whose process died (killed, machine restarted) is reported once by the next command, with the
  command that resumes it. Under WSL, closing the last WSL window can shut the Linux VM down after a
  while, which stops the job the same way.
- The job's output goes to `index.sqlite-job.log` next to the index.

## What gets indexed

- Text: `.md .markdown .mdx .txt .text .rst .org .adoc .asciidoc .tex .wiki .html .htm`, plus whatever
  you add with `--ext`. Text is split into chunks of about 384 tokens with 48 tokens of overlap, at line
  boundaries where possible, and results show the first line of the chunk (`notes.md:12`; `--json` also
  has `line_end`).
- PDF: the text layer of each page, chunked per page. Results show page numbers (`p.3`). Pages with
  almost no words, such as a figure page whose text layer is only a page number, are left out. Scanned
  PDFs without a text layer are skipped and listed by `anygrep status --problems`.
- Images: `.jpg .jpeg .png .webp .bmp .gif` (first frame). EXIF orientation is applied, and transparency
  is flattened onto white. Images smaller than 32 px are skipped.

### Skipped files

- Hidden files and folders.
- Anything matched by `.gitignore`, `.ignore` or `.anygrepignore` files. Nested files work like in git,
  including `!` rules, and the files in the folders above an indexed folder apply up to its git
  repository root. A folder that those rules exclude is still indexed when you name it, with a note.
- `node_modules`, `__pycache__`, virtualenvs, symlinks, binary files and empty files.
- Files over the size limits: 20 MB text, 300 MB PDF, 80 MB image.
- Files of other types, including Office and other document formats that are not supported yet:
  `.docx .doc .xlsx .xls .pptx .ppt .odt .ods .odp .rtf .epub`. When a run (`anygrep index`, or the
  indexing before a search) finds new files to index, it also says how many of these documents it
  skipped, in one line on stderr such as `anygrep: skipped 12 documents (.docx, .xlsx); these formats
  are not supported yet`, so they are not taken for missing from the results. A run with nothing new
  does not repeat it, unless the folder has nothing to index apart from such documents: then every
  run says so. `-q` leaves it out, and the results and exit status do not change. Other unsupported
  files (code, archives, media, Apple Pages, Numbers and Keynote files) are not counted.

Files that cannot be read are recorded and retried after they change. Files that failed because of
permissions or a model error, for example running out of memory, are retried on every run. Folders that
cannot be read are reported, and files indexed there before are kept. `anygrep status --problems` lists
skipped and failed files.

Paths are stored with symlinks resolved, so the same folder reached through a symlink, or typed with
different letter case on macOS or Windows, is not indexed twice.

## Searching

```
anygrep [OPTIONS] QUERY [PATH ...]
anygrep [OPTIONS] -e QUERY [PATH ...]       # QUERY starts with '-' or is a command name
anygrep [OPTIONS] --image FILE [PATH ...]   # search with an image
anygrep [OPTIONS] -Q [PATH ...]             # interactive: the model stays loaded
```

QUERY is one argument, so quote a query of several words; the arguments after it are PATHs. If a PATH
does not exist, anygrep stops and asks whether you forgot the quotes. A first argument that is exactly a
command name (`index`, `status`, `remove`, `rm`, `reset`, `model`, `help`) runs that command; to search
for such a word use `anygrep -e status`, `anygrep -- status` or `anygrep search status`.

Results are limited to files at or under the PATHs (default: the current folder), so `anygrep QUERY` in
`~/notes/recipes` searches only the recipes. `--everywhere` searches every indexed folder instead.

Options:

- Scope: `-g GLOB` / `--glob GLOB` keeps paths matching the glob, `-g '!GLOB'` drops them (repeatable;
  see "Filtering by path" below). `-t TYPE` / `--type TYPE` keeps only `text`, `pdf` or `image`
  (repeatable or a comma list: `-t pdf,image`), and `-T TYPE` / `--type-not TYPE` drops one.
- Indexing before the search: `--no-index` searches the index as it is and writes nothing (the index is
  opened read-only). When stderr is not a terminal, `-y` / `--yes` indexes everything first and
  `--index-budget S` sets the time budget in seconds (default 30, or `ANYGREP_INDEX_BUDGET`; 0 allows
  only the cheap updates).
- Results: `-m N` / `--max-count N` is the total number of results (default 10; not per file as in
  grep). `--weak` also shows results below the relevance cutoff. `--chunks` shows every matching chunk,
  not one per file. `--no-collapse` lists near-duplicate files separately.
- Output: `-l` / `--files-with-matches` prints file paths only, `-0` / `--null` ends them with NUL
  (with `-l`), `--json` writes JSON Lines, `-q` / `--quiet` prints nothing (exit status only),
  `--color auto|always|never` (`NO_COLOR` is honoured), `--open` opens the top result.
- `-n` and `-i` are accepted for grep habits and change nothing: line numbers are always shown, and
  semantic search ignores case anyway.

### Indexing before a search

Before it searches, anygrep looks for new, changed, moved and deleted files under the PATHs:

- Deleted, moved and touched files are always updated; that is quick.
- When stderr is a terminal, all new and changed files are indexed first, with a progress bar (count,
  percent, elapsed and remaining time), then anygrep searches. It does not ask first. Ctrl-C stops it
  with status 130; finished files are kept and the next search or `anygrep index` continues.
- When stderr is not a terminal (a script, cron), they are indexed first only when the internal
  estimate is within the budget (30 seconds by default; it learns from earlier runs, and `anygrep
  status` shows "measured speed"). Otherwise nothing is indexed, all or nothing. If some files under
  the PATH are indexed, anygrep searches those and then says that PATH has new files, with the command
  to update it (`anygrep index PATH`). If none are, it prints that command and exits with status 2.
- A PATH outside the indexed folders becomes one when it is indexed this way, with a note that says how
  to undo it (`anygrep rm PATH`).
- Your home folder, its parent and a drive root are never walked automatically: anygrep searches what is
  indexed under them, or says to name a folder in them. Drive roots include drives and volumes mounted
  under `/mnt`, `/media` and `/Volumes`; under WSL `/mnt/c`, `/mnt/c/Users` and the Windows home folder
  `/mnt/c/Users/<name>` count as well. When stderr is not a terminal, the scan also
  stops after 20,000 entries or 2 seconds; a folder that is partly indexed is then searched as it is,
  with a note, and one with nothing indexed is not indexed (status 2, with the command to run once).
- While another anygrep process is indexing, a search does not index and searches the index as it is.
  When that process is a background job (`anygrep index --background`), the search ends with the job's
  progress line.

### Output

On a terminal, results are ranked with their score, location and a snippet:

```
 1  0.764  notes/sourdough.md:1
      # Sourdough bread, the version that finally worked  Starter: feed 1:1:1 ...
 2  0.712  docs/employee-handbook.pdf  p.3  [pdf]
      Vacation and leave  New employees receive 25 paid vacation days ...
```

When the output is piped or redirected, each result is one grep-style line, `PATH:LINE:SNIPPET`. LINE is
always a number: the first line of the chunk for text, `1` for PDFs and images, whose snippet starts with
`[p.N]` or is `[image]`. That works with `fzf --delimiter :` and `vim -q`:

```
notes/sourdough.md:1:# Sourdough bread, the version that finally worked Starter: feed 1:1:1 ...
docs/employee-handbook.pdf:1:[p.3] Vacation and leave New employees receive 25 paid vacation days ...
screenshots/error-dialog.png:1:[image]
```

`--json` writes one JSON object per result and line, in rank order, with the keys `rank`, `path`
(absolute), `type`, `line_start`, `line_end`, `page`, `score`, `weak`, `exact`, `snippet` and `similar` (the
collapsed near-duplicates, with the same keys except `rank` and `similar`). `-l` prints each matching file
once, best first.

```console
$ anygrep -l 'tax receipts'
notes/finance/tax-checklist.md
notes/finance/kakutei-shinkoku.md
```

Everything that is not a result (notes, hints, progress, errors) goes to stderr.

Piped or redirected output is UTF-8. On Windows it uses the console code page instead, which is what
PowerShell and cmd decode it with; set `[Console]::OutputEncoding = [Text.Encoding]::UTF8` or
`PYTHONUTF8=1` to get UTF-8 there. When the output is not UTF-8, `--json` writes non-ASCII characters as
`\uXXXX` escapes so nothing is lost.

### Exit status

| Status | Meaning |
|---|---|
| 0 | at least one result passed the relevance cutoff or matched exactly (any result with `--weak`) |
| 1 | the scope was searched and nothing passed the cutoff |
| 2 | an error, including a missing PATH or nothing indexed under the PATH |
| 130 | interrupted (Ctrl-C) |

Other commands exit with 0 or 2 (130 when interrupted). `anygrep index` exits with 2 while a file
under its PATHs has failed to index (a broken image or PDF, say), after indexing the rest. A broken
file is tried again only when it changes, but it counts on every run until then (the summary says
"N still failed from an earlier run"). Skipped files (too small, no text layer) do not count.
`anygrep status --problems` lists both.

### Weak results

Search hides results whose score is below the cutoff for their type (see [relevance.md](relevance.md)).
`--weak` shows them too, marked `(weak)`, and then they count as matches for the exit status. On a
terminal, when results were left out with a score just under the cutoff (less than 0.02 below it),
the list ends with a dim line such as `2 more results just below the cutoff (--weak shows them)`, so
such a file is not taken for missing. Results further below are not counted. The count also covers
up to 50 results past `-m`; when some of them would not fit in `-m`, the line says which `-m` shows
them (`1 more result just below the cutoff (--weak -m 4 shows it)`), and when more may lie past those
50 it says "at least" (`at least 3 more results ... (--weak and a larger -m show them)`). PATHs,
`-g`, `-t` and the grouping of near-duplicates apply as for the results shown. When nothing passes
the cutoff, anygrep says "No strong matches; showing the closest results"
and lists the closest three, or `-m` of them (at most 10) when you give `-m`, marked as weak; the exit
status is still 1. Piped output, `-l` and `--json` show weak results only with `--weak` (in `--json`
with `"weak": true`); otherwise they print nothing and stderr says
`anygrep: no strong matches (use --weak to see the closest)`.

```console
$ anygrep 'qwzx plorb fnord' -m 2; echo "exit=$?"
No strong matches; showing the closest results.
 1  0.628  photos/whiteboard.jpg  [image]  (weak)
 2  0.615  screenshots/terminal.png  [image]  (weak)
exit=1
```

### Exact matches

A text query of 3 or more characters is also looked up as a plain string in the indexed text and in
the file names (the name only, not the folders above it). Case is ignored for every script, and text is
compared in Unicode NFKC form, so full-width letters and digits match plain ones (`ＡＢＣ１２３` matches
`abc123`) and accents match however they are encoded. Files that contain it are always listed, even when
their score is below the cutoff, and come before all other results, best score first. On a terminal they
are marked `(exact)`, in `--json` they have `"exact": true` (and `"weak": false`, whatever the score),
and the piped format and `-l` simply list them first. They count as matches for the exit status. PATHs,
`-g`, `-t` and `-m` apply to them as to any result.

This is for things that have no meaning to match, such as an invoice number, an error code, a part
number or a file name: `anygrep INV-20391` finds the one note that mentions it. It is a substring
match, not a regular expression, and the whole query must appear as typed (`tax return` does not match
`tax\nreturn` across a line break). Because it is a substring match, a longer query also matches inside
words (`budget` in `budgeting`). Two limits keep common words from flooding the results:

- A query of 3 or 4 ASCII letters matches only whole words: `run` finds "run the script" but not
  "truncates", and `pip` does not match "pipeline". A letter or digit of any script continues the
  word, so `run` does not match "runé", "érun" or "run2". The underscore and the hyphen separate
  words, so it matches "run_script" and "run-script", and so do Chinese, Japanese and Korean
  characters, which are written without spaces next to Latin words ("runを実行"). Text is folded
  first, so full-width "ＲＵＮ" counts as `run`. File names follow the same rule.
- A query of plain words (letters of any script, spaces, apostrophes and hyphens inside a word, as in
  `e-mail` or `follow-up`) found in more than 10 of the files
  searched, and in more than 5% of them, is no clue to a particular file. Such a query (`the`,
  `and`, a word every note uses) gets no exact matches and is searched by meaning only. In a scope
  of 3 or more files, the same goes for plain words found in more than half of them (`the` in a
  folder of five notes).
- A query with a digit or any other character, such as an ID, an error code or a file name
  (`INV-2026-0412`, `v2.14`, `report.pdf`), is always matched, however many files contain it: the
  same order number in 12 related files still tells them from the rest.

Shorter queries (1 or 2 characters, which includes many Japanese words such as `税金`) are searched by
meaning only. Text inside an image is not matched unless it is in the file name. Searching an image
never matches exactly.

Exact matching uses SQLite's FTS5 full-text index with the trigram tokenizer, which needs SQLite 3.34 or
later (Python's own `sqlite3` module has had it for years on most systems). Without it anygrep searches
by meaning only, and `anygrep status` says so. An index made by an earlier version of anygrep gets the
full-text table on the next `anygrep index` or search that indexes; until then, exact matches are found
by reading the stored text, which is slower on a large index. Once this version has written to an
index, anygrep 0.2 and earlier refuse it ("created by a newer anygrep"), because they would change it
without keeping the full-text table.

### Near-duplicates

Near-duplicate files are shown once: exact copies (the same bytes), and notes or PDFs whose embeddings
and words are almost the same (an exported or slightly edited version, the same note in two folders).
Images are grouped only when they are exact copies; a resized or cropped screenshot, or a chart with
other numbers, is a result of its own. The best-ranked one is listed with a note, and the next results
fill up to `-m` distinct ones:

```console
$ anygrep "marketing budget cut"     # in the test corpus of scripts/measure_duplicates.py
 1  0.748  base/notes/meetings/2026-03-budget.md:1
      ...12 Attendees: Mia, Jonas, Priya, me ## Decisions - Marketing budget for Q2 is cut by 15 percent...
      (+1 similar: copies/drafts/2026-03-budget-v2.md)
 2  0.640  heldout-mixed/projects/imgshrink/README.md:1
      # imgshrink Batch-resize and recompress photos from the command line. pip install imgshrink...
```

`--no-collapse` lists them separately, and so do the piped format and `-l`, which have no room for the
note. With `--json` each result has a `similar` list with the score,
path and position of every collapsed file. `--chunks` does not collapse. The limits of this grouping are
in [relevance.md](relevance.md#near-duplicates), and the rule and how it was measured are in
[DESIGN.md](DESIGN.md) (in Japanese).

### Filtering by path

PATHs limit a search to folders or files. `-g` takes a folder prefix (`-g ~/notes/work`) or a glob
matched against the full path or its end (`-g '*.md'`, `-g 'meetings/*'`). A glob that starts with `~`,
`./`, `../` or `/` is matched against the full path from that folder (`-g './notes/*.md'`). With `!` in
front it excludes (`-g '!*.pdf'`). When several include globs are given, a path must match one of them;
it must match none of the excludes.

## Other commands

- `anygrep status`: index location, size, counts per type, image budget, model files. `--problems` lists
  skipped or failed files; `--json` for machine-readable output.
- `anygrep remove PATH ...` (or `anygrep rm`): remove folders or files from the index.
- `anygrep reset`: delete the index. The model files and the coordination lock file are kept.
  `-y` skips the confirmation.
- `anygrep model [--download]`: show the model files or download them now.

## Offline use and mirrors

Everything runs locally and anygrep has no telemetry. The only network access is the one-time model
download from huggingface.co. It is pinned to a fixed revision and checked against SHA-256 hashes.

onnxruntime has its own usage telemetry: the 1.31 Linux build sends it to Microsoft, and the Windows
build logs it through ETW. anygrep switches it off before loading the model.

- `HF_ENDPOINT` points the download at a mirror.
- To work fully offline, run `anygrep model --download` once on a connected machine, or copy the files
  listed by `anygrep model` into the directory it shows, or set `ANYGREP_MODEL_DIR` to where they are.
- With `ANYGREP_OFFLINE=1` anygrep never tries the network and tells you which files are missing.

## Limitations

- No OCR. Small text inside images is often missed; `--image-tokens 280` helps. Scanned PDFs are skipped.
- Image indexing is slow on a CPU (see [benchmarks.md](benchmarks.md)). It is not a photo library
  replacement.
- Each search reloads the model; use `anygrep -Q` for several queries.
- Search is brute force in memory. It was measured at 100,000 chunks (about 0.4 s per search after the model loads).
- The relevance cutoff was tuned on small synthetic corpora. Use `--weak` if a result is missing.
- No audio, video or Office documents (`.docx`, `.xlsx`, `.pptx` and the like are skipped with a note).
- No file watching: run `anygrep index` again, for example from cron.

## Demo corpus

To try anygrep on a small synthetic corpus (12 notes, 3 PDFs, 8 images), from a clone of this repository:

```sh
python scripts/make_demo_corpus.py ~/anygrep-demo/demo-corpus
cd ~/anygrep-demo/demo-corpus
anygrep "screenshot of an error about disk space" --db ~/anygrep-demo/index.sqlite
```

Generate it outside the clone: anygrep honours `.gitignore`, and the repository ignores generated demo
files.
