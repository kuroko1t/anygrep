# anygrep

[![PyPI](https://img.shields.io/pypi/v/anygrep)](https://pypi.org/project/anygrep/)
[![Python](https://img.shields.io/pypi/pyversions/anygrep)](https://pypi.org/project/anygrep/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

anygrep searches your notes, PDFs and screenshots by meaning, from the terminal. It runs one model
([EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), int8 ONNX) on the CPU and keeps one
local index. Everything stays on your machine; the only download is the model.

![anygrep demo](demo/anygrep.gif)

## Install

```sh
pipx install anygrep        # or: uv tool install anygrep
uvx anygrep "when is the tax return due" ~/notes   # or try it without installing
```

Requires Python 3.10+. Tested on Linux, Windows 11 and macOS (Apple silicon, in CI). On an Intel Mac
use Python 3.10-3.13 (`pipx install --python python3.13 anygrep`); on Windows on ARM use Python 3.11+.
The model files (about 540 MB) are downloaded on first use.

To uninstall, run `pipx uninstall anygrep` and delete the index and model folders that `anygrep status`
shows ([details](docs/usage.md#uninstalling)).

## Usage

```sh
anygrep "wifi password reset"                 # search here (new files are indexed first)
anygrep "when is the tax return due" ~/notes  # search one folder
anygrep --image whiteboard.jpg ~/Pictures     # find similar pictures
anygrep -l "invoice from acme"                # file names only; piped output is path:line:text
anygrep -Q ~/notes                            # interactive: the model stays loaded
anygrep index --background ~/Pictures         # big folder: index in a low-priority process
```

Queries and documents can be in different languages. Weak matches are hidden (`--weak` shows them) and
near-duplicate files are shown once. See `anygrep -h` and [docs/usage.md](docs/usage.md).

The first search in a big folder is slow because it indexes the new files first, with a progress bar on
a terminal. Ctrl-C keeps what is done and the next run continues. `index --background` does it while you
work (`anygrep status` shows progress). Your home folder is never indexed automatically; name a folder
inside it.

It reads text files (`.md .txt .html` and more), the text layer of PDFs, and `.jpg .png .webp .bmp
.gif` images. Hidden, ignored and binary files are skipped.

## Use from Python

```python
import anygrep

with anygrep.Index() as idx:  # the index the command uses
    idx.update("~/notes")  # index new and changed files
    for hit in idx.search("when is the tax return due", k=5):
        print(hit.score, hit.path, hit.snippet)
    for hit in idx.search(image="whiteboard.jpg", type="image"):  # search by image too
        print(hit.path)
```

Install it with pip in your project's environment. It uses the same index, scores and filtering as the
command and prints nothing. The API may change before 1.0; see [docs/python-api.md](docs/python-api.md).

## Performance

On a Ryzen 7 7700 with the default 8 threads, text indexes at about 1,100 tokens/s and an image takes
about 1.2 s. A search command takes about 1.5 s, mostly loading the model. `-j 12` is about 17% faster
on an idle 16-thread CPU. More in [docs/benchmarks.md](docs/benchmarks.md).

Text chunks, PDF pages and images share one vector space in a single SQLite file, and a search compares
the query with them by brute-force cosine similarity ([how it works](docs/how-it-works.md),
[relevance](docs/relevance.md), [comparison](docs/comparison.md)).

## Limitations

- No OCR: small text inside images is often missed, and scanned PDFs are skipped.
- Image indexing is slow on a CPU. This won't replace a photo library.
- No audio, video or Office documents, and no file watching (run `anygrep index` from cron).

## Privacy

Everything runs locally. The only network access is the one-time model download, checked against hashes
([offline use](docs/usage.md#offline-use-and-mirrors)), and anygrep has no telemetry. onnxruntime has its
own usage telemetry (the 1.31 Linux build sends it to Microsoft, the Windows build logs it through ETW);
anygrep turns it off before loading the model.

## License

MIT. The model is not part of this package. anygrep downloads
[an int8 ONNX conversion](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) of EmbeddingGemma 2,
which Google releases under the Apache License 2.0. Its model card also refers to the Gemma Prohibited
Use Policy, and your use of the model is subject to those terms. anygrep is not affiliated with or
endorsed by Google.
