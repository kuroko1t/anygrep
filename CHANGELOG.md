# Changelog

All notable changes to anygrep are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
