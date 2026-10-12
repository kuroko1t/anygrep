# Changelog

All notable changes to anygrep are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- The index records the processing generation each file was indexed with. When a new version of
  anygrep changes how some kind of file is extracted, chunked or embedded, the next update (`anygrep
  index`, the indexing before a search, `Index.update()`) indexes those files again, with progress and
  resumable like any other change, instead of needing `anygrep reset`. A search that does not update
  them says so in one line on stderr; `anygrep status` shows the count and the Python API has
  `Index.update_pending`. A different model revision still needs a reset.
- Exact matches. A text query of 3 or more characters is also looked up as a plain string in the
  indexed text and the file names, ignoring case and Unicode width (NFKC), so an invoice number, an
  error code or a file name finds the file that contains it. Those files always come first, best score
  first, even below the relevance cutoff, marked `(exact)` on a terminal, `"exact": true` in `--json`
  and `Hit.exact` in the Python API. Queries of 3 or 4 ASCII letters match whole words only (a letter or digit of
  any script continues a word; `_`, `-` and Chinese, Japanese or Korean characters do not), and a
  query of plain words found in more than 10 of the files searched (and more than 5% of them), or
  in more than half of a scope of 3 or more files, gives no exact matches. A query with a digit or
  another symbol (an ID, an error code, a file name) is always matched.
  It uses an SQLite FTS5 trigram index (SQLite 3.34 or later; without it searches use meaning only and
  `anygrep status` says so). Existing indexes get it on the next update, without re-embedding anything;
  the index file grows by about a third to a half (47% in a test with 2,000 notes).
- On a terminal, a search that hid results just below the cutoff ends with one dim line such as `2 more
  results just below the cutoff (--weak shows them)`. It also counts up to 50 results past `-m` and
  then names the `-m` that shows them, or says "at least" when more may follow. Piped output, `-l`,
  `--json` and the exit status do not change.

- When a run finds new files to index (`anygrep index` or the indexing before a search), it says in
  one line on stderr how many files in a common document format it skipped (`.docx`, `.xlsx`,
  `.pptx`, `.odt`, `.rtf`, `.epub` and the like), since those are not supported yet. `-q` leaves it
  out; stdout and the exit status do not change. The Python API has
  `UpdateResult.unsupported_documents`.

### Changed

- The index schema is now 2. An index of schema 1 is read as it is and moved to schema 2 by the first
  update; anygrep 0.2 and earlier then refuse it and ask for an upgrade, because they would change the
  chunks without keeping the exact-match table.
- Images are grouped as near-duplicates only when they are exact copies (the same bytes). Before, images
  whose embeddings were almost the same were grouped too, which also hid charts or statements whose
  numbers differ. A resized or recompressed copy of a picture is now listed as a result of its own.
  Notes and PDFs are grouped as before.
- `anygrep index` exits with status 2 while a file under its PATHs has failed to index (it still
  indexes the rest and prints the summary). That includes a file that failed in an earlier run and
  was not tried again because it has not changed; the summary counts it as "still failed from an
  earlier run". Skipped files do not count. Before, it exited with 0.

### Fixed

- `index --stop` no longer kills a process that only reuses the PID of a background job that died: the
  job records its process start time and is killed only when it matches. When that cannot be checked,
  `--stop` fails instead of killing.
- `index --stop` right after `index --background` (before the index file exists) stops the job
  instead of saying none is running.
- `index --stop` while a background job is still starting cancels it: its worker exits without
  indexing. Before, the stop was reported but the job then ran anyway. A worker also never writes
  to the state of a job that is not its own.
- A background job's start time is read in a fixed time zone on macOS, so a shell with another `TZ`
  no longer takes a running job for a dead one.
- Python API: `type=["PDF", "Image"]` is accepted like `type="PDF, Image"`.
- Python API: calling the same `Index` from the `progress` callback of `update()` raises `RuntimeError`.
  A search from there could record images at the wrong token budget, and an update from there released
  the writer lock early.
- Python API: `k`, `threads` and `image_tokens` must be whole numbers (`image_tokens=70.0` was stored
  as `"70.0"` and broke later searches); `type` takes a comma list like `-t`; a PIL `image` gets its
  EXIF orientation applied like a file; searching an empty index no longer loads the model.

## [0.1.0] - 2026-10-12

First public release.

### Added

- `anygrep QUERY [PATH ...]` searches Markdown and text notes, PDF pages (text layer) and images
  (`.jpg .jpeg .png .webp .bmp .gif`) by meaning, in one ranked list. PATH defaults to the current
  folder; `--everywhere` searches every indexed folder. `--image FILE` searches with an image, and `-Q`
  keeps the model loaded for several queries.
- A search indexes new and changed files under its PATHs first. On a terminal it indexes everything with
  a progress bar (Ctrl-C keeps what is done). When stderr is not a terminal it indexes only if that
  should take about 30 seconds or less (`--index-budget`, `ANYGREP_INDEX_BUDGET`, `-y`, `--no-index`).
- `anygrep index [PATH ...]` indexes ahead of time, and `index --background PATH` does it in a detached,
  low-priority process (`anygrep status` shows progress, `index --stop` stops it). Indexing is
  incremental and resumable, honours `.gitignore`, `.ignore` and `.anygrepignore`, and moved or renamed
  files keep their embeddings.
- grep-style output: `path:LINE` results, one `PATH:LINE:SNIPPET` line per result when piped, `-l`,
  `-0`, `-q`, `--json` (JSON Lines), `--color`, `--open`, plus `-g/--glob`, `-t/--type`,
  `-T/--type-not`, `-m/--max-count`, `--chunks` and `-e`. Exit status 0 found, 1 nothing relevant,
  2 error, 130 interrupted.
- Embeddings from EmbeddingGemma 2 (int8 ONNX conversion from onnx-community) on the CPU through ONNX
  Runtime, without PyTorch. Image preprocessing is a numpy port of the reference processor. Image token
  budget `--image-tokens 70|140|280` (default 140).
- A relevance cutoff per pair of query and result type, set with a margin above a bundled reference set,
  so weak matches are hidden; `--weak` shows them.
- Near-duplicate collapsing (exact copies, exported or slightly edited versions, resized screenshots)
  using word signatures stored at index time. Short templated documents such as monthly statements
  usually stay separate; long ones that differ only in a few numbers can still be grouped.
  `--no-collapse` turns it off.
- `anygrep status`, `remove` (`rm`), `reset` and `model [--download]`. The model download is pinned,
  hash-checked and resumable; `ANYGREP_OFFLINE=1` for fully offline use.
- onnxruntime's own usage telemetry is switched off.
- A Python API: `anygrep.Index` with `update()` and `search()`, the `Hit` and `UpdateResult`
  dataclasses and an `AnygrepError` exception hierarchy ([docs/python-api.md](docs/python-api.md)).
  It may change before 1.0.

[Unreleased]: https://github.com/kuroko1t/anygrep/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kuroko1t/anygrep/releases/tag/v0.1.0
