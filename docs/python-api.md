# Python API

anygrep can be imported as well as run. `anygrep.Index` uses the same index file, model, scores,
relevance cutoff and duplicate grouping as the command. It prints nothing and never asks anything.

The API is new and may change before 1.0. Pin the version if you depend on it.

```sh
pip install anygrep   # into your project's environment; a pipx install is not importable
```

```python
import anygrep

with anygrep.Index() as idx:
    result = idx.update("~/notes")  # index new and changed files
    print(result.new, "new files")
    for hit in idx.search("when is the tax return due", k=5):
        print(f"{hit.score:.3f} {hit.path}:{hit.line_start} {hit.snippet}")
    for hit in idx.search(image="whiteboard.jpg", type="image"):
        print(hit.path)
```

`import anygrep` is cheap: numpy and onnxruntime are imported when an `Index` is opened or used, and
the model is loaded by the first `update()` or `search()` that needs it, then kept until `close()`.

## `anygrep.Index(path=None, *, threads=None, create=True)`

Opens an index. If the file does not exist, it is created, along with any missing folders above it.

- `path`: the index file. The default is the one the `anygrep` command uses (`$ANYGREP_DB`, or
  `index.sqlite` in the user data folder), so the command and your code see the same files. An empty
  string raises `ValueError` instead of picking the default.
- `threads`: CPU threads for the model. The default is the command's: `$ANYGREP_THREADS`, or half the
  logical CPUs up to 8.
- `create=False`: raise `FileNotFoundError` when the file does not exist instead of creating an empty
  index. Use it in code that only searches, so a mistyped path is an error rather than no results.

`Index` is a context manager; `close()` closes the file and drops the model, and calling it twice is
fine. One `Index` can be shared between threads, for example by the handlers of a web app. Their calls
take turns, and the model is loaded only once.

If the index file is deleted or replaced while an `Index` is open, for example by `anygrep reset`, the
next call opens the file that is there now (creating an empty one, unless `create=False`).

Attributes: `path` (the resolved index file), `threads`, and `roots`, the list of indexed folders and
files.

## `Index.update(paths=None, *, image_tokens=None, images=True, extensions=(), progress=None)`

Like `anygrep index`. Indexes new and changed files under `paths` (one path or a list of folders and
files), drops deleted ones and reuses the vectors of moved files. The paths become indexed folders.
Without `paths` (`None`), every indexed folder is updated; an empty list raises `ValueError`. A folder that is missing (say, on an unmounted drive)
is skipped and keeps its entries.

- `image_tokens`: 70, 140 or 280 vision tokens per image. Changing it re-embeds the images already
  indexed. By default the index keeps the budget it has (140 for a new index).
- `images=False`: skip images in this run. Images already indexed stay.
- `extensions`: also index these extensions as plain text, for example `[".py", ".json"]`. They are
  remembered for later runs, as with `--ext`.
- `progress`: called as `progress(done, total)` with the number of files finished and to do, first
  with `done == 0`. It may be called more than once with the same numbers.

Returns an `UpdateResult` with the counts of files `new`, `updated`, `unchanged`, `removed`, `moved`,
`skipped` and `failed`, and `seconds`. `anygrep status --problems` lists the skipped and failed files.

Finished files are saved as they go. If the run stops early, because of Ctrl-C or an exception raised
from `progress`, that exception is raised and the next `update()` continues where it stopped.

Only one process changes an index at a time. When another one does, for example a background index
started with `anygrep index --background`, `update()` raises `IndexBusyError` at once. Searching works
meanwhile.

The first call that needs the model downloads it (about 540 MB) without any output. Run
`anygrep model --download` beforehand to see the progress, or set `ANYGREP_OFFLINE=1` to make sure
nothing is downloaded.

## `Index.search(query=None, *, image=None, k=10, paths=None, glob=None, type=None, weak=False, collapse=True, chunks=False)`

Searches by meaning and returns a list of `Hit`, best first. Give either a text `query` or `image`, a
file path or a PIL image. When `image` is a file in the index, that file is left out of the results.

- `k`: the most results to return (`-m`). There are often fewer, because weak matches are left out.
- `paths`: search only under these folders or files. The default (`None`) is the whole index; the
  command searches the current folder by default. An empty list raises `ValueError` rather than
  searching everything.
- `glob`: one pattern or a list, as with `-g`: `"*.md"`, `"notes/*"`, `"!archive/*"` to exclude. An
  empty list raises `ValueError`.
- `type`: `"text"`, `"pdf"`, `"image"` or a list of them, as with `-t`.
- `weak=True`: also return results below the relevance cutoff, marked with `Hit.weak`
  ([relevance.md](relevance.md)).
- `collapse=False`: list near-duplicate files separately instead of under the best one
  ([usage.md](usage.md#near-duplicates)).
- `chunks=True`: return the best chunks rather than the best files, so a file can appear more than
  once.

`search()` only reads the index; unlike the command it does not index new files first. Call `update()`
for that. When nothing passes the cutoff the list is empty, and so it is for a new or empty index; the
command's "closest results" fallback is `search(..., weak=True, k=3)`.

The results are the ones `anygrep --json --everywhere` prints for the same options, with the same
order, `weak` flags and grouping. Two things differ in form: `--json` rounds `score` to 4 decimals,
and it has `null` where `Hit.snippet` is `""` (images).

## `Hit`

A frozen dataclass.

| Field | |
| --- | --- |
| `path` | absolute path of the file |
| `type` | `"text"`, `"pdf"` or `"image"` |
| `score` | cosine similarity, about 0 to 1; higher is closer |
| `weak` | below the relevance cutoff (only with `weak=True`) |
| `page` | page number for a PDF, else `None` |
| `line_start`, `line_end` | 1-based lines of the chunk in a text file, else `None` |
| `snippet` | one line around the query words, as in `--json` (`""` for images, where `--json` has `null`) |
| `text` | the whole text of the matching chunk (`None` for images) |
| `similar` | near-duplicate files grouped under this one, as `Hit`s |

## Errors

All errors anygrep raises on purpose derive from `anygrep.AnygrepError`:

- `IndexBusyError`: another anygrep process is changing the index.
- `IndexAccessError`: the index file cannot be opened or written: the path is a folder, the file or
  its folder is read-only, or the disk is full. It is also an `OSError`. A read-only index can still
  be searched.
- `IndexMismatchError`: the file is not an anygrep index, or was built by a different version or
  model; `anygrep reset` and index again.
- `ModelUnavailableError`: the model files are missing and cannot be downloaded (offline, or a network
  error).
- `ModelLoadError`: the model files are there but the installed onnxruntime or tokenizers cannot load
  them.

Wrong arguments, including an invalid `$ANYGREP_THREADS`, raise `ValueError`, and paths that do not exist raise `FileNotFoundError`. An image
that cannot be read raises what Pillow raises (an `OSError`), or `ValueError` when it is too small or too large.

`anygrep.__version__` is the installed version.
