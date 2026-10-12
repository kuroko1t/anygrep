# Contributing

Bug reports, search results that look wrong, and small pull requests are all welcome.

## Reporting a problem

Please use the bug report template and include the output of `anygrep --version` and `anygrep status`.
If a search result looks wrong, `anygrep "your query" --weak --json -m 10` shows the scores and
which results were hidden as weak, which is usually what is needed to understand it. Remove anything
private from paths and snippets before posting.

## Development

```sh
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                 # the real-model test runs only if the model files are present
ruff check . && ruff format --check .
```

`anygrep model --download` fetches the model files (about 540 MB) so that the real-model tests run too.

To try a change by hand without touching your own index, point anygrep at a scratch index and data
folder: `ANYGREP_DB=/path/to/scratch.sqlite ANYGREP_HOME=/path/to/scratch-data anygrep ...`. A
background job (`anygrep index --background`) keeps its state and log next to that index.

Speed changes should keep the results identical: compare `--json` output and, for images, the stored
vectors before and after (`tests/test_speed.py` has examples).

Changes to the relevance cutoff or the near-duplicate rule should come with the output of
`scripts/calibrate_thresholds.py` or `scripts/measure_duplicates.py` (and `scripts/heldout_corpus.py`),
so the effect on the measured corpora is visible. The design notes are in `docs/DESIGN.md`.
