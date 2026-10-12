"""The Python interface: `anygrep.Index`.

    import anygrep

    with anygrep.Index() as idx:
        idx.update("~/notes")
        for hit in idx.search("when is the tax return due", k=5):
            print(hit.path, hit.score, hit.snippet)

It uses the same index, model, scores, relevance cutoff and duplicate
collapsing as the command line. Nothing is printed and nothing is asked. The
heavy modules (numpy, onnxruntime) are imported when an Index is used, not
when anygrep is imported, and the model is loaded on the first call that
needs it.
"""

from __future__ import annotations

import functools
import numbers
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Concatenate, ParamSpec, TypeVar

from .config import IMAGE_BUDGETS, default_index_path, default_threads
from .errors import IndexAccessError, IndexBusyError, IndexMismatchError
from .extract import KINDS

if TYPE_CHECKING:
    from PIL import Image

    from .embedder import Embedder
    from .indexer import IndexOptions
    from .store import Hit as _StoreHit
    from .store import Store

StrPath = str | os.PathLike[str]

# The snippet width of `anygrep --json`.
SNIPPET_COLS = 240

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _locked(method: Callable[Concatenate[Index, _P], _R]) -> Callable[Concatenate[Index, _P], _R]:
    """Run the method while holding the Index's lock: threads take turns."""

    @functools.wraps(method)
    def wrapper(self: Index, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._mutex:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass(frozen=True)
class Hit:
    """One search result. `type` is "text", "pdf" or "image"; `page` is set
    for PDFs and `line_start`/`line_end` (1-based) for text files. `snippet`
    is a one-line excerpt around the query words ("" for images, where
    --json has null) and `text`
    the whole matching chunk. `weak` marks a result below the relevance
    cutoff (only returned with `weak=True`). `exact` marks a file whose
    text or file name contains the query as typed (ignoring case and
    Unicode width); these come first and are never marked weak, even below
    the cutoff. `similar` lists the near-duplicate files collapsed under this
    one."""

    path: str
    type: str
    score: float
    weak: bool = False
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    snippet: str = ""
    text: str | None = field(default=None, repr=False)
    similar: tuple[Hit, ...] = field(default=(), repr=False)
    exact: bool = False


@dataclass(frozen=True)
class UpdateResult:
    """What Index.update() did, in files."""

    new: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    moved: int = 0
    skipped: int = 0
    failed: int = 0
    seconds: float = 0.0


class Index:
    """An anygrep index: the SQLite file the command line uses.

    `path` is the index file; by default the one `anygrep` uses
    ($ANYGREP_DB, or index.sqlite in the user data folder). It is created,
    with any missing parent folders, when it does not exist, unless `create`
    is false; then a missing file raises FileNotFoundError. `threads` is the
    number of CPU threads for the model (default: $ANYGREP_THREADS, or half
    the CPU threads up to 8).

    Use it as a context manager or call close(). One Index can be shared by
    several threads; their calls take turns.
    """

    def __init__(
        self, path: StrPath | None = None, *, threads: int | None = None, create: bool = True
    ) -> None:
        if threads is None:
            try:
                threads = default_threads()
            except ValueError:
                value = os.environ.get("ANYGREP_THREADS")
                raise ValueError(
                    f"ANYGREP_THREADS must be a whole number of at least 1, got {value!r}"
                ) from None
        else:
            threads = _whole(threads, "threads")
            if threads < 1:
                raise ValueError("threads must be at least 1")
        if path is None:
            self.path = default_index_path().resolve()
        elif not os.fspath(path):
            raise ValueError("path is empty; leave it out to use the default index")
        else:
            self.path = Path(path).expanduser().resolve()
        self.threads = threads
        self.create = create
        self._mutex = threading.RLock()
        self._updating = False  # update() runs: its progress callback may not re-enter
        self._load: Callable[[bool], Embedder] | None = None
        self._store: Store | None = None
        self._store, self._file = self._open()

    def _open(self) -> tuple[Store, tuple[int, int] | None]:
        from .store import Store

        where = f"the index at {self.path}"
        if self.path.is_dir():
            raise IndexAccessError(f"cannot open {where}: it is a folder")
        if not self.create and not self.path.exists():
            raise FileNotFoundError(f"no index at {self.path}")
        try:
            store = Store(self.path, shared=True)
        except sqlite3.OperationalError as e:
            raise IndexAccessError(f"cannot open {where}: {e}") from None
        except sqlite3.DatabaseError as e:
            raise IndexMismatchError(f"{self.path} is not an anygrep index ({e}).") from None
        except OSError as e:  # creating the file or its folder
            raise IndexAccessError(f"cannot open {where}: {e.strerror or e}") from None
        return store, _file_id(self.path)

    # -- lifetime
    @_locked
    def close(self) -> None:
        """Close the index file and drop the model. Calling it again does
        nothing."""
        self._not_from_progress()
        store, self._store = self._store, None
        self._load = None
        if store is not None:
            store.close()

    def __enter__(self) -> Index:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._store is None else "open"
        return f"<anygrep.Index {str(self.path)!r} ({state})>"

    def _db(self) -> Store:
        if self._store is None:
            raise ValueError("the index is closed")
        if _file_id(self.path) != self._file:
            # Deleted (`anygrep reset`) or replaced since it was opened: use
            # the file that is there now, so nothing is written to a file
            # that no longer has a name.
            store, self._file = self._open()
            old, self._store = self._store, store
            old.close()
        return self._store

    def _not_from_progress(self) -> None:
        # Only the thread running update() gets past the lock while it runs:
        # a call from its progress callback.
        if self._updating:
            raise RuntimeError("cannot use this Index from its progress callback")

    def _embedder(self, vision: bool) -> Embedder:
        if self._load is None:
            from .embedder import embedder_factory

            self._load = embedder_factory(self.threads, quiet=True)
        return self._load(vision)

    # -- reading
    @property
    @_locked
    def roots(self) -> list[str]:
        """The indexed folders and files, as update() without paths sees them."""
        return self._db().roots()

    @property
    @_locked
    def update_pending(self) -> bool:
        """True when some indexed files were processed by an older version of
        anygrep (a fix to text extraction, chunking or the like has come
        since). They are still searched as they are; the next update()
        processes them again. False does not mean the index is complete:
        new, changed, failed or partly indexed files do not count."""
        return self._db().count_stale() > 0

    @_locked
    def search(
        self,
        query: str | None = None,
        *,
        image: StrPath | Image.Image | None = None,
        k: int = 10,
        paths: StrPath | Iterable[StrPath] | None = None,
        glob: str | Iterable[str] | None = None,
        type: str | Iterable[str] | None = None,
        weak: bool = False,
        collapse: bool = True,
        chunks: bool = False,
    ) -> list[Hit]:
        """Search by meaning with a text `query` or an `image` (a file or a
        PIL image), best first. At most `k` results; often fewer, because
        results below the relevance cutoff are left out unless `weak` is
        true.

        `paths` limits the search to these folders or files (default: the
        whole index). `glob` patterns work like `-g` ("*.md", "!drafts/*")
        and `type` like `-t` ("text", "pdf", "image", "pdf,image"). An
        `image` given as a PIL image is turned by its EXIF orientation, as a
        file is. Near-duplicate files
        are grouped under the best one (Hit.similar) unless `collapse` is
        false. With `chunks`, a file can appear more than once.

        Only what is indexed is searched: call update() first to pick up new
        or changed files.
        """
        from PIL import ImageOps

        from .embedder import _downscale, load_image, to_rgb
        from .paths import canonical, search_scope
        from .search import clean_text, image_query, query_budget, scope_matcher, search

        self._not_from_progress()
        if query is None and image is None:
            raise ValueError("give a text query or image=")
        if query is not None and image is not None:
            raise ValueError("give either a text query or image=, not both")
        if query is not None and not clean_text(query).strip():
            raise ValueError("the query is empty")
        k = _whole(k, "k")
        if k < 1:
            raise ValueError("k must be at least 1")
        kinds = _kinds(type)
        scope_paths = _path_list(paths, "search the whole index")
        _check_exist(scope_paths)
        prefixes = search_scope(scope_paths) if scope_paths else []
        globs = [] if glob is None else [glob] if isinstance(glob, str) else list(glob)
        if glob is not None and not globs:
            raise ValueError("glob is an empty list; leave it out to match every file")
        scope = scope_matcher(prefixes, globs)
        store = self._db()

        exclude = qimage = image_queries = None
        if isinstance(image, (str, os.PathLike)):
            # Read before the model is loaded; the file itself is not a result.
            exclude = str(canonical(image))
            qimage = load_image(Path(image).expanduser())
        elif image is not None:
            small = _downscale(image)
            qimage = to_rgb(ImageOps.exif_transpose(small) or small)  # a copy if turned
        if not store.has_chunks() or (prefixes and not store.has_files_under(prefixes)):
            return []  # nothing to search: no need for the model
        budget = query_budget(store)
        embedder = self._embedder(qimage is not None)
        old_budget, embedder.image_budget = embedder.image_budget, budget
        try:
            if qimage is not None:
                qvec, image_queries = image_query(
                    store, embedder, qimage, budget, kinds, prefixes, scope, exclude
                )
                text = None
            else:
                text = clean_text(query or "")
                qvec = embedder.embed_query(text)
        finally:
            embedder.image_budget = old_budget
        hits = search(
            store,
            qvec,
            n=k,
            kinds=kinds,
            scope=scope,
            per_file=not chunks,
            exclude_path=exclude,
            query_kind="text" if text is not None else "image",
            budget=budget,
            hide_weak=not weak,
            collapse=collapse,
            image_queries=image_queries,
            prefixes=prefixes,
            exact=text,
        )
        return [_public_hit(h, text) for h in hits]

    # -- writing
    @_locked
    def update(
        self,
        paths: StrPath | Iterable[StrPath] | None = None,
        *,
        image_tokens: int | None = None,
        images: bool = True,
        extensions: Iterable[str] = (),
        progress: Callable[[int, int], object] | None = None,
    ) -> UpdateResult:
        """Index new and changed files under `paths` (folders or files) and
        drop deleted ones, like `anygrep index`. The paths become indexed
        folders. Without paths, every indexed folder is updated.

        `image_tokens` (70, 140 or 280) changes the image token budget and
        re-embeds the images already indexed; by default the index keeps its
        budget (140 for a new one). `images=False` skips images in this run.
        `extensions` (".py", ".json") are also indexed as plain text from now
        on. `progress(done, total)` is called as files finish; it may not
        call this Index (that raises RuntimeError).

        Finished files are saved as they go. If the run is interrupted
        (Ctrl-C, or an exception raised from `progress`), that exception is
        raised and the next update() continues where it stopped.

        Raises IndexBusyError when another anygrep process, such as a
        background index, is changing the index, IndexAccessError when the
        index cannot be written, and FileNotFoundError for a path that does
        not exist.
        """
        self._not_from_progress()
        try:
            self._updating = True
            return self._update(paths, image_tokens, images, extensions, progress)
        except sqlite3.OperationalError as e:
            msg = str(e)
            if "readonly" in msg:
                reason = "it is read-only"
            elif "full" in msg:
                reason = "the disk is full; finished files are kept"
            elif "unable to open" in msg:
                reason = "its folder cannot be written"
            else:
                raise
            raise IndexAccessError(f"cannot write the index at {self.path}: {reason}") from None
        except PermissionError as e:  # the lock file next to the index
            raise IndexAccessError(
                f"cannot write the index at {self.path}: {e.strerror or e}"
            ) from None
        finally:
            self._updating = False

    def _update(
        self,
        paths: StrPath | Iterable[StrPath] | None,
        image_tokens: int | None,
        images: bool,
        extensions: Iterable[str],
        progress: Callable[[int, int], object] | None,
    ) -> UpdateResult:
        from . import job
        from .indexer import IndexOptions, index_settings, index_targets, update_index

        if image_tokens is not None:
            image_tokens = _whole(image_tokens, "image_tokens")
            if image_tokens not in IMAGE_BUDGETS:
                raise ValueError(f"image_tokens must be one of {IMAGE_BUDGETS}")
        exts = [extensions] if isinstance(extensions, str) else list(extensions)
        targets = _path_list(paths, "update every indexed folder")
        _check_exist(targets)
        store = self._db()
        if not store.try_write_lock(probe_db=False):
            raise IndexBusyError(self._busy_message())
        try:
            if job.check(self.path, clean=False)[0] is not None:
                # A background job was started and has not taken the lock yet.
                raise IndexBusyError(self._busy_message())
            if not targets and not store.roots():
                return UpdateResult()
            budget, _previous, text_exts = index_settings(store, image_tokens, images, exts)
            roots, dropped, _missing = index_targets(store, targets)
            opts = IndexOptions(
                image_budget=budget, images=images, extra_text_exts=frozenset(text_exts), quiet=True
            )
            watch = _Progress(progress, opts) if progress is not None else None
            try:
                rep = update_index(store, roots, opts, self._embedder, dropped=dropped)
            except KeyboardInterrupt:
                if watch is not None and watch.error is not None:
                    raise watch.error from None
                raise
        finally:
            store.release_write_lock()
        if rep.interrupted:
            if watch is not None and watch.error is not None:
                raise watch.error
            raise KeyboardInterrupt
        return UpdateResult(
            new=rep.new,
            updated=rep.updated,
            unchanged=rep.unchanged,
            removed=rep.removed,
            moved=rep.moved,
            skipped=rep.skipped,
            failed=rep.errors,
            seconds=rep.seconds,
        )

    def _busy_message(self) -> str:
        from . import job

        if job.check(self.path, clean=False)[0] is not None:
            return (
                f"a background index of {self.path} is running; wait for it "
                "(anygrep status) or stop it (anygrep index --stop)"
            )
        return f"another anygrep process is updating {self.path}; try again when it is done"


class _Progress:
    """Calls progress(done, total) from the indexer's hooks. An exception
    raised by it stops the run like Ctrl-C (the indexer keeps the finished
    files and would otherwise count the exception as a failed file); update()
    raises it afterwards."""

    def __init__(self, progress: Callable[[int, int], object], opts: IndexOptions) -> None:
        self.progress = progress
        self.total = 0
        self.error: BaseException | None = None
        opts.on_plan = self.on_plan
        opts.on_progress = self.on_progress

    def _call(self, done: int) -> None:
        try:
            self.progress(done, self.total)
        except BaseException as e:
            self.error = e
            raise KeyboardInterrupt from e

    def on_plan(self, todo: list) -> list:
        self.total = len(todo)
        self._call(0)
        return todo

    def on_progress(self, done: int, cost: float, seconds: float, kind: str, end: bool) -> None:
        self._call(done)


def _path_list(paths: StrPath | Iterable[StrPath] | None, default: str) -> list[str]:
    """The paths as strings. None means everything; an empty list is an
    error rather than everything, so a computed list that comes out empty
    does not widen the scope."""
    if paths is None:
        return []
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = [os.path.expanduser(os.fspath(p)) for p in paths]
    if not out:
        raise ValueError(f"paths is an empty list; leave it out to {default}")
    if not all(out):
        raise ValueError("paths contains an empty string")
    return out


def _file_id(path: Path) -> tuple[int, int] | None:
    """What identifies the file at `path` (None: there is none), to notice
    when it is deleted or replaced."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_dev, st.st_ino


def _check_exist(paths: list[str]) -> None:
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"no such file or directory: {p!r}")


def _kinds(value: str | Iterable[str] | None) -> list[str] | None:
    """The file types to search (None: all)."""
    if value is None:
        return None
    # "pdf,image" as with -t.
    if isinstance(value, str):
        kinds = {v.strip().lower() for v in value.split(",") if v.strip()}
    else:
        kinds = {str(v).strip().lower() for v in value}
    bad = kinds - set(KINDS)
    if bad or not kinds:
        raise ValueError(
            f'type must be one or more of {", ".join(KINDS)}, e.g. "pdf" or ["pdf", "image"]'
        )
    return sorted(kinds) if kinds != set(KINDS) else None


def _whole(value: object, name: str) -> int:
    """`value` as an int: whole numbers only (not bool, not 2.0)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a whole number, got {value!r}")
    return int(value)


def _public_hit(h: _StoreHit, query: str | None) -> Hit:
    from .search import snippet

    return Hit(
        path=h.path,
        type=h.kind,
        score=h.score,
        weak=h.weak,
        page=h.page,
        line_start=h.line_start,
        line_end=h.line_end,
        snippet=snippet(h.text, query, SNIPPET_COLS),
        text=h.text,
        similar=tuple(_public_hit(x, query) for x in h.similar),
        exact=h.exact,
    )
