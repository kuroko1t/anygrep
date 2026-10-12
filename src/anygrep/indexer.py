"""Incremental, resumable indexing.

Change detection uses (size, mtime); when only the mtime changed the content
hash decides. Finished files are committed in small transactions, so an
interrupted run (Ctrl-C, crash, power loss) keeps everything done so far and
the next run continues with the remaining files.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import sys
import time
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from .chunking import chunk_text
from .config import CHUNK_OVERLAP, CHUNK_TOKENS, DEFAULT_IMAGE_BUDGET
from .embedder import Embedder, ImageTooLarge, load_image
from .errors import ModelLoadError
from .extract import SkipFile, kind_of, read_pdf_pages, read_text_file
from .paths import canonical, sha256_file
from .signature import SignatureBuilder
from .store import ChunkData, FileRow, Store
from .walker import FileEntry, WalkStats, walk

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

# Rough cost model (seconds at 4 threads), used for the progress bar ETA and,
# scaled by the measured throughput factor, for the search's indexing budget.
# Measured at 4 threads: 1.52 s per image at budget 140 (0.56x of that at 70,
# 2.1x at 280).
IMAGE_COST = {70: 0.85, 140: 1.5, 280: 3.2}
TEXT_TOKENS_PER_SEC = 800.0
MIN_IMAGE_SIDE = 32
# PDF pages with fewer letters than this (a page number, a lone label) are skipped.
MIN_PAGE_LETTERS = 8
TEXT_BATCH_CHUNKS = 32
# A file with more chunks than this is embedded and committed in slices of
# TEXT_BATCH_CHUNKS: the progress bar moves, and an interrupted run keeps the
# slices done (the file stays 'partial' and the next run continues it).
LARGE_FILE_CHUNKS = 128
PARTIAL_NOTE = "indexing was interrupted; the next run continues it"
COMMIT_EVERY_SEC = 3.0
# Measured seconds / estimated seconds of past runs, stored in the index meta
# as "throughput_factor". Only runs that embedded at least CALIB_MIN_SECONDS
# of work update it (an exponential moving average); before any measurement
# a conservative default is used.
CALIB_KEY = "throughput_factor"
CALIB_DEFAULT = 1.5
CALIB_RANGE = (0.5, 4.0)
CALIB_ALPHA = 0.5
CALIB_MIN_SECONDS = 20.0
# Background runs (fewer threads, low priority) store their measured speed
# under their own key, so they do not make the foreground estimate (the 30 s
# budget decision of a search) pessimistic.
BG_CALIB_KEY = "bg_throughput_factor"
# The progress bar of `anygrep index`, of a search that indexes first, and of
# a background job (anygrep status): count, percent, elapsed<remaining.
BAR_FORMAT = "{desc} {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]"
# Errors that depend on the environment rather than on the file content (a
# permission fixed with chmod does not change the mtime). Files that failed
# with one of these are retried on every run.
RETRY_ERRORS = (
    "PermissionError",
    "FileNotFoundError",
    "TimeoutError",
    "InterruptedError",
    "BlockingIOError",
    "MemoryError",
    # The model failed on this file (for example ONNX Runtime could not
    # allocate memory); the file itself may be fine.
    "EmbedError",
)


@dataclass
class IndexOptions:
    image_budget: int = DEFAULT_IMAGE_BUDGET
    images: bool = True
    extra_text_exts: frozenset[str] = frozenset()
    max_size: dict[str, int] = field(default_factory=dict)
    chunk_tokens: int = CHUNK_TOKENS
    chunk_overlap: int = CHUNK_OVERLAP
    quiet: bool = False
    # Seconds before the progress bar appears (0: at once).
    progress_delay: float = 0.0
    # Background index hooks. should_stop() is asked once per folder while
    # walking (True raises KeyboardInterrupt). on_plan(todo) may reorder the
    # files to index and returns them. on_progress(files_done, cost_done,
    # seconds, kind, burst_end) is called after every file and every slice of
    # a large file; burst_end is false for all but the last of the files a
    # text batch finishes at once (the state worth saving). It may raise
    # KeyboardInterrupt to stop the run; should_stop is also asked between
    # model runs inside a batch.
    should_stop: Callable[[], bool] | None = None
    on_plan: Callable[[list], list] | None = None
    on_progress: Callable[[int, float, float, str, bool], None] | None = None


@dataclass
class IndexReport:
    scanned: int = 0
    new: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    moved: int = 0  # moved or renamed files whose index rows were reused
    reused: int = 0  # images whose vector was copied from an identical image
    reused_seconds: float = 0.0
    # Estimated cost of the files embedded (not copied) and the seconds from
    # the start of embedding (with the model load): what calibrates.
    embedded_cost: float = 0.0
    embed_seconds: float = 0.0
    skipped: int = 0
    errors: int = 0
    deferred_images: int = 0
    seconds: float = 0.0
    interrupted: bool = False
    by_kind_seconds: dict[str, float] = field(default_factory=dict)
    by_kind_files: dict[str, int] = field(default_factory=dict)
    walk: WalkStats = field(default_factory=WalkStats)


def throughput_factor(store: Store, key: str = CALIB_KEY) -> float:
    """The stored measured/estimated time ratio, or the default."""
    try:
        v = float(store.get_meta(key) or CALIB_DEFAULT)
    except ValueError:
        v = CALIB_DEFAULT
    lo, hi = CALIB_RANGE
    return min(hi, max(lo, v)) if v == v else CALIB_DEFAULT


def update_throughput(
    store: Store, estimated: float, measured: float, key: str = CALIB_KEY
) -> float | None:
    """Fold one run's measured/estimated ratio into the stored factor; None
    when the run was too short to say anything."""
    if estimated <= 0 or measured < CALIB_MIN_SECONDS:
        return None
    lo, hi = CALIB_RANGE
    sample = min(hi, max(lo, measured / estimated))
    old = store.get_meta(key)
    new = (
        sample
        if old is None
        else CALIB_ALPHA * sample + (1 - CALIB_ALPHA) * throughput_factor(store, key)
    )
    new = min(hi, max(lo, new))
    store.set_meta(key, f"{new:.4f}")
    return new


def estimate_cost(e: FileEntry, budget: int) -> float:
    if e.kind == "image":
        return IMAGE_COST.get(budget, 1.5)
    tokens = min(e.size, 20 * 2**20) / 60 if e.kind == "pdf" else e.size / 4
    return 0.02 + tokens / TEXT_TOKENS_PER_SEC


def index_order(todo: list[tuple[FileEntry, bool]]) -> list[tuple[FileEntry, bool]]:
    """Text, then PDF (by path), then images newest first: an interrupted or
    still running index has made the cheap, most searched files findable."""
    order = {"text": 0, "pdf": 1, "image": 2}
    return sorted(
        todo,
        key=lambda t: (
            order[t[0].kind],
            -t[0].mtime_ns if t[0].kind == "image" else 0,
            str(t[0].path),
        ),
    )


@dataclass
class _Pending:
    entry: FileEntry
    sha: str | None = None
    status: str = "ok"
    note: str | None = None
    chunks: list[ChunkData] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)  # awaiting embedding
    cost: float = 0.0
    cost_done: float = 0.0  # progress bar share already shown
    sig: bytes | None = None  # word signature (text and PDF)
    is_new: bool = True
    seconds: float = 0.0
    reused: bool = False  # an image whose vector was copied


def _should_retry(status: str, note: str | None) -> bool:
    return status == "error" and bool(note) and note.split(":", 1)[0] in RETRY_ERRORS


def plan(
    store: Store,
    roots: list[Path],
    opts: IndexOptions,
    report: IndexReport,
    max_entries: int | None = None,
    max_seconds: float | None = None,
    apply: bool = True,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[tuple[FileEntry, bool]], list[int]]:
    """Return (files to (re)index with is_new flag, file ids to delete).

    Unchanged files whose metadata changed are touched and moved files are
    re-keyed, unless `apply` is false (nothing is written then). With
    max_entries or max_seconds the walk raises walker.WalkTruncated at that
    cap, before anything is written."""
    known = frozenset(Path(r) for r in store.roots())
    found = walk(
        roots, opts.extra_text_exts, opts.max_size, report.walk, known_roots=known,
        max_entries=max_entries, max_seconds=max_seconds, should_stop=should_stop,
    )  # fmt: skip
    entries = list(found)
    report.scanned = len(entries)
    existing = {}
    for r in roots:
        for row in store.files_under(str(r)):
            existing[row.path] = row
    seen: set[str] = set()
    todo: list[tuple[FileEntry, bool]] = []
    touched: list[tuple[int, int, int]] = []
    for e in entries:
        p = str(e.path)
        seen.add(p)
        if e.kind == "image" and not opts.images:
            report.deferred_images += p not in existing
            continue
        row = existing.get(p)
        if row is None:
            todo.append((e, True))
            continue
        if row.status == "partial":  # interrupted while indexing: continue it
            todo.append((e, True))
            continue
        if row.kind != e.kind or _should_retry(row.status, row.note):
            todo.append((e, False))
            continue
        if e.kind == "image" and row.status == "ok" and row.budget != opts.image_budget:
            todo.append((e, False))
            continue
        if row.size == e.size and row.mtime_ns == e.mtime_ns:
            report.unchanged += 1
            continue
        if row.size == e.size and row.sha256:
            try:
                same = sha256_file(e.path) == row.sha256
            except OSError:
                same = False
            if same:
                touched.append((row.id, e.size, e.mtime_ns))
                report.unchanged += 1
                continue
        todo.append((e, False))
    # One transaction for all of them: after a checkout or a sync that touched
    # thousands of files, one commit per file was most of the run.
    if apply:
        store.touch_files(touched)
    # Keep rows of files that were not seen only because a folder could not be
    # listed this time, or because of a --max-size limit given for this run.
    # Rows under another indexed root nested in this one belong to that root
    # (it may be indexed despite the ignore rules of this one).
    keep = report.walk.held
    blocked = [str(d) for d in report.walk.unreadable_dirs]
    for root in report.walk.missing_roots:
        if str(root) not in existing:
            # A directory vanished after root validation (e.g. an unmounted
            # drive). A standalone file has an exact row and remains removable.
            blocked.append(str(root))
            report.walk.unreadable_dirs.append(root)
    blocked += [str(k) for k in known if k not in roots and any(r in k.parents for r in roots)]
    blocked_prefixes = tuple(d.rstrip("/\\") + os.sep for d in blocked)

    def is_blocked(p: str) -> bool:
        return p in keep or p in blocked or p.startswith(blocked_prefixes)

    gone = [row for p, row in existing.items() if p not in seen and not is_blocked(p)]
    todo = _reuse_moved(store, todo, gone, opts, report, apply)
    removed = [row.id for row in gone]
    return index_order(todo), removed


def _reuse_moved(
    store: Store,
    todo: list[tuple[FileEntry, bool]],
    gone: list,
    opts: IndexOptions,
    report: IndexReport,
    apply: bool = True,
) -> list[tuple[FileEntry, bool]]:
    """Move the index rows of files that were moved or renamed instead of
    embedding them again. A new file takes over a vanished file's row when
    the content hash matches and the embedding would be the same: images (at
    the same budget), or text and PDFs whose file name (part of the embedded
    title) did not change. Matched rows are taken out of `gone`."""
    cands: dict[tuple[str, int], list] = {}
    for row in gone:
        same_budget = row.kind != "image" or row.budget == opts.image_budget
        if row.status == "ok" and row.sha256 and same_budget:
            cands.setdefault((row.kind, row.size), []).append(row)
    if not cands:
        return todo
    keep: list[tuple[FileEntry, bool]] = []
    moves: list[tuple[FileEntry, FileRow]] = []
    for e, is_new in todo:
        rows = cands.get((e.kind, e.size)) if is_new else None
        if rows and (e.kind == "image" or any(Path(r.path).stem == e.path.stem for r in rows)):
            try:
                sha = sha256_file(e.path)
            except OSError:
                sha = None
            row = next(
                (
                    r
                    for r in rows
                    if r.sha256 == sha and (e.kind == "image" or Path(r.path).stem == e.path.stem)
                ),
                None,
            )
            if row is not None:
                moves.append((e, row))
                rows.remove(row)
                gone.remove(row)
                continue
        keep.append((e, is_new))
    if moves and not apply:
        report.moved += len(moves)
    elif moves:
        with store.transaction():
            for e, row in moves:
                if store.move_file(row.id, str(e.path), e.size, e.mtime_ns):
                    report.moved += 1
                else:  # removed by another process meanwhile: index it anew
                    keep.append((e, True))
                    gone.append(row)
    return keep


def run_index(
    store: Store,
    roots: list[Path],
    opts: IndexOptions,
    get_embedder: Callable[[bool], Embedder],
    planned: tuple[list[tuple[FileEntry, bool]], list[int], IndexReport] | None = None,
    add_roots: bool = True,
    calibrate: str | None = CALIB_KEY,
) -> IndexReport:
    """Index `roots` (they become indexed roots unless add_roots is false).
    `planned` is the result of a plan() call already made for them:
    (todo, removed, report). `calibrate` is the meta key the measured speed
    is stored under (None: not stored)."""
    t_start = time.perf_counter()
    roots = list(dict.fromkeys(canonical(r) for r in roots))
    if add_roots:
        store.add_roots(
            str(r)
            for r in roots
            if r.is_dir() or (r.is_file() and kind_of(r, opts.extra_text_exts) is not None)
        )
    if planned is None:
        report = IndexReport()
        if not opts.quiet:
            print("Scanning files...", file=sys.stderr)
        todo, removed = plan(store, roots, opts, report, should_stop=opts.should_stop)
    else:
        todo, removed, report = planned
    if opts.on_plan is not None:
        todo = opts.on_plan(todo)
    if removed:
        report.removed = store.delete_files(removed)
    if not todo:
        report.seconds = time.perf_counter() - t_start
        return report
    t_embed = time.perf_counter()

    need_vision = any(e.kind == "image" for e, _ in todo)
    embedder = get_embedder(need_vision)
    embedder.image_budget = opts.image_budget
    # A stop request is seen between model runs, not only between batches.
    embedder.should_stop = opts.should_stop
    t_rate = time.perf_counter()  # progress speed: from after the model load

    total_cost = sum(estimate_cost(e, opts.image_budget) for e, _ in todo)
    bar = tqdm(
        total=total_cost,
        disable=opts.quiet,
        bar_format=BAR_FORMAT,
        desc=f"Indexing 0/{len(todo)}",
        file=sys.stderr,
        dynamic_ncols=True,
        delay=opts.progress_delay,
    )
    done_files = 0
    done_cost = 0.0  # estimated cost of the finished files
    reused_cost = 0.0  # estimated cost of the images copied, not embedded
    burst = False  # finishing the files of one batch: one bar/progress update
    held = 0.0  # bar progress held back during a burst
    queue: list[_Pending] = []  # text/pdf waiting for embedding
    ready: list[_Pending] = []  # embedded, waiting for commit
    last_commit = time.perf_counter()

    def advance(p: _Pending, cost: float) -> None:
        nonlocal held
        cost = min(cost, p.cost - p.cost_done, bar.total - bar.n - held)
        if cost > 0:
            p.cost_done += cost
            if burst:
                held += cost
            else:
                bar.update(cost)

    def progress(kind: str, extra: float = 0.0, end: bool = True) -> None:
        if opts.on_progress is not None:
            secs = time.perf_counter() - t_rate
            opts.on_progress(done_files, done_cost + extra, secs, kind, end)

    def finish_all(ps: list[_Pending]) -> None:
        """Finish the files a batch embedded at once: the bar and the
        progress hook see them as one step, not as their first file."""
        nonlocal burst, held
        if not ps:
            return
        burst = True
        try:
            for i, p in enumerate(ps):
                finish(p, end=i == len(ps) - 1)
        finally:
            burst = False
            if held > 0:
                bar.update(held)
            held = 0.0

    def finish(p: _Pending, stored: bool = False, end: bool = True) -> None:
        """A file is done; `stored` when its rows are already committed."""
        nonlocal done_files, done_cost
        if stored:
            count(p)
        else:
            ready.append(p)
        done_files += 1
        k = p.entry.kind
        # Only embedded files count, so s/file is not lowered by skipped or
        # copied ones.
        if p.reused:
            report.reused_seconds += p.seconds
        else:
            report.by_kind_files[k] = report.by_kind_files.get(k, 0) + (p.status == "ok")
            report.by_kind_seconds[k] = report.by_kind_seconds.get(k, 0.0) + p.seconds
        bar.set_description_str(f"Indexing {done_files}/{len(todo)}", refresh=False)
        advance(p, p.cost)
        done_cost += p.cost
        progress(k, end=end)

    def count(p: _Pending) -> None:
        if p.status == "ok":
            if p.is_new:
                report.new += 1
            else:
                report.updated += 1
        elif p.status == "skipped":
            report.skipped += 1
        else:
            report.errors += 1

    def embed_batch(batch: list[_Pending]) -> None:
        texts = [t for p in batch for t in p.texts]
        titles = [p.entry.path.stem for p in batch for _ in p.texts]
        t0 = time.perf_counter()
        vecs = embedder.embed_documents(texts, titles)
        dt = time.perf_counter() - t0
        n_tok = sum(len(t) for t in texts) or 1
        i = 0
        for p in batch:
            for c in p.chunks:
                c.vec = vecs[i]
                i += 1
            p.seconds += dt * sum(len(t) for t in p.texts) / n_tok
            p.texts = []

    def flush_queue() -> None:
        if any(p.texts for p in queue):
            try:
                embed_batch(queue)
            except ModelLoadError:
                raise
            except Exception:
                # Find the file that broke the batch; the others are kept.
                for p in queue:
                    try:
                        embed_batch([p])
                    except ModelLoadError:
                        raise
                    except Exception as ex:
                        _mark_error(p, ex, prefix="EmbedError: ")
        done = list(queue)
        queue.clear()
        finish_all(done)

    def commit() -> None:
        nonlocal last_commit
        if not ready:
            return
        with store.transaction():
            for p in ready:
                e = p.entry
                fid = store.put_file(
                    str(e.path), e.kind, e.size, e.mtime_ns, p.sha,
                    opts.image_budget if e.kind == "image" else None,
                    p.status, p.note, p.chunks, p.sig if p.status == "ok" else None,
                )  # fmt: skip
                if e.kind == "image" and p.status == "ok" and p.sha is not None:
                    known_images[p.sha] = fid
                    run_images.pop(p.sha, None)
                count(p)
        ready.clear()
        last_commit = time.perf_counter()

    def index_large(p: _Pending) -> None:
        """Embed and commit a file with many chunks slice by slice. A file
        left 'partial' by an interrupted run with the same content continues
        after its last stored chunk."""
        e = p.entry
        flush_queue()
        commit()
        t0 = time.perf_counter()
        assert p.sha is not None
        resume = store.resume_point(str(e.path), e.size, e.mtime_ns, p.sha)
        start = 0
        if (
            resume is not None
            and 0 < resume[1] <= len(p.chunks)
            and p.chunks[resume[1] - 1].text == resume[2]
        ):
            fid, start = resume[0], resume[1]
        if start == 0:
            with store.transaction():
                fid = store.put_file(
                    str(e.path), e.kind, e.size, e.mtime_ns, p.sha, None,
                    "partial", PARTIAL_NOTE, [], p.sig,
                )  # fmt: skip
        n = len(p.chunks)
        advance(p, p.cost * start / n)
        title = e.path.stem
        for i in range(start, n, TEXT_BATCH_CHUNKS):
            part = p.chunks[i : i + TEXT_BATCH_CHUNKS]
            try:
                vecs = embedder.embed_documents([c.text or "" for c in part], [title] * len(part))
            except ModelLoadError:
                raise
            except Exception as ex:
                p.seconds += time.perf_counter() - t0
                _mark_error(p, ex, prefix="EmbedError: ")
                finish(p)  # replaces the partial row
                commit()
                return
            for c, v in zip(part, vecs, strict=True):
                c.vec = v
            with store.transaction():
                store.add_chunks(fid, part)
            for c in part:  # stored: free the memory
                c.vec = None
                c.text = None
            advance(p, p.cost * len(part) / n)
            progress(e.kind, p.cost * min(n, i + len(part)) / n)
        with store.transaction():
            store.set_status(fid, "ok", None)
        p.seconds += time.perf_counter() - t0
        p.texts, p.chunks = [], []
        finish(p, stored=True)

    # Identical images (same bytes, same budget) get the same vector, whatever
    # their names: an image already indexed, or embedded earlier in this run,
    # is copied instead of embedded again. Images are embedded one at a time:
    # running several through the vision encoder together was not faster and
    # took much more memory.
    known_images = store.image_shas(opts.image_budget) if need_vision else {}
    run_images: dict[str, np.ndarray] = {}  # embedded, not committed yet

    def known_vector(sha: str) -> np.ndarray | None:
        if sha in run_images:
            return run_images[sha]
        fid = known_images.get(sha)
        return None if fid is None else store.image_vector(fid, sha, opts.image_budget)

    def image_done(p: _Pending, vec: np.ndarray) -> None:
        p.chunks = [ChunkData(0, vec)]
        assert p.sha is not None
        run_images[p.sha] = vec
        finish(p)

    def embed_one(p: _Pending, im: Image.Image) -> None:
        t0 = time.perf_counter()
        try:
            vec = embedder.embed_image(im)
        except ModelLoadError:
            raise
        except ValueError as ex:
            _mark_error(p, ex)
        except Exception as ex:
            _mark_error(p, ex, prefix="EmbedError: ")
        else:
            p.seconds += time.perf_counter() - t0
            image_done(p, vec)
            return
        p.seconds += time.perf_counter() - t0
        finish(p)

    try:
        for e, is_new in todo:
            p = _Pending(e, cost=estimate_cost(e, opts.image_budget), is_new=is_new)
            t0 = time.perf_counter()
            try:
                p.sha = sha256_file(e.path)
                if e.kind == "image":
                    vec = known_vector(p.sha)
                    if vec is not None:
                        report.reused += 1
                        reused_cost += p.cost
                        p.reused = True
                        p.seconds = time.perf_counter() - t0
                        image_done(p, vec)
                    else:
                        try:
                            im = load_image(e.path)
                        except ImageTooLarge as ex:
                            raise SkipFile(str(ex)) from None
                        if min(im.size) < MIN_IMAGE_SIDE:
                            raise SkipFile(f"image smaller than {MIN_IMAGE_SIDE}px")
                        p.seconds = time.perf_counter() - t0
                        embed_one(p, im)
                else:
                    _extract_chunks(p, embedder, opts)
                    p.seconds = time.perf_counter() - t0
                    if not p.texts:
                        p.status, p.note = (
                            "skipped",
                            ("no text layer (scanned PDF?)" if e.kind == "pdf" else "no text"),
                        )
                        finish(p)
                    elif len(p.texts) > LARGE_FILE_CHUNKS:
                        index_large(p)
                    else:
                        queue.append(p)
            except SkipFile as s:
                p.status, p.note, p.chunks, p.texts = "skipped", str(s), [], []
                finish(p)
            except (KeyboardInterrupt, ModelLoadError):
                raise
            except Exception as ex:
                _mark_error(p, ex)
                finish(p)
            if queue and (
                sum(len(q.texts) for q in queue) >= TEXT_BATCH_CHUNKS
                or time.perf_counter() - last_commit >= COMMIT_EVERY_SEC
            ):
                flush_queue()
            if ready and time.perf_counter() - last_commit >= COMMIT_EVERY_SEC:
                commit()
        flush_queue()
        commit()
    except KeyboardInterrupt:
        report.interrupted = True
        commit()  # keep finished files; queued ones are redone next run
    except sqlite3.Error:
        raise  # the index cannot be written (full disk, read-only): do not retry
    except BaseException:
        # Keep the files finished so far, then report the original error.
        with contextlib.suppress(Exception):
            flush_queue()
        with contextlib.suppress(Exception):
            commit()
        raise
    finally:
        bar.close()
    report.seconds = time.perf_counter() - t_start
    # Copied images take no model time: only the embedded ones calibrate.
    report.embedded_cost = done_cost - reused_cost
    report.embed_seconds = time.perf_counter() - t_embed
    if not report.interrupted and calibrate:
        with contextlib.suppress(sqlite3.Error):
            update_throughput(store, total_cost - reused_cost, report.embed_seconds, calibrate)
    return report


def index_settings(
    store: Store, image_tokens: int | None, images: bool, extensions: Iterable[str] = ()
) -> tuple[int, int | None, set[str]]:
    """Settings of an `index` run: (image budget, the budget of the images
    indexed so far or None before the first run, text extensions). The
    budget is `image_tokens`, or the one an interrupted budget change left,
    or the stored one; a change is recorded as pending until update_index()
    finishes it. `extensions` are added to the remembered ones."""
    stored = store.get_meta("image_budget")
    stored_budget = int(stored or DEFAULT_IMAGE_BUDGET)
    # A budget change that was interrupted is finished by a plain `index`.
    pending = store.get_meta("image_budget_pending")
    budget = image_tokens or (int(pending) if pending else stored_budget)
    if images:
        if budget != stored_budget:
            store.set_meta("image_budget_pending", str(budget))
        elif pending:
            store.delete_meta("image_budget_pending")
    return budget, int(stored) if stored else None, text_extensions(store, extensions)


def text_extensions(store: Store, extensions: Iterable[str] = ()) -> set[str]:
    """The extensions indexed as plain text: the remembered ones and
    `extensions` (".py" or "py")."""
    exts = set(filter(None, (store.get_meta("text_exts") or "").split(",")))
    for e in extensions:
        exts.add(e.lower() if e.startswith(".") else "." + e.lower())
    return exts


def index_targets(
    store: Store, paths: Iterable[str | os.PathLike[str]] | None
) -> tuple[list[Path], int, list[Path]]:
    """(roots to index, files dropped, missing folders skipped). Without
    `paths`, every indexed folder and file: a file indexed on its own that
    was deleted is dropped from the index; a missing folder is only skipped
    (it may be on a drive that is not mounted right now)."""
    if paths:
        return [Path(p).expanduser() for p in paths], 0, []
    roots = [Path(r) for r in store.roots()]
    gone = [r for r in roots if not r.exists()]
    gone_files = [r for r in gone if store.get_file(str(r)) is not None]
    dropped = 0
    if gone_files:
        dropped = store.delete_files(store.get_file(str(r)).id for r in gone_files)
        store.remove_roots(str(r) for r in gone_files)
    gone_dirs = [r for r in gone if r not in gone_files]
    return [r for r in roots if r.exists()], dropped, gone_dirs


def update_index(
    store: Store,
    roots: list[Path],
    opts: IndexOptions,
    get_embedder: Callable[[bool], Embedder],
    dropped: int = 0,
    calibrate: str | None = CALIB_KEY,
) -> IndexReport:
    """`anygrep index` after index_settings() and index_targets(), with the
    writer lock held: remember the text extensions, index `roots` and record
    the image budget once every image has it."""
    store.set_meta("text_exts", ",".join(sorted(opts.extra_text_exts)))
    rep = run_index(store, roots, opts, get_embedder, calibrate=calibrate)
    rep.removed += dropped
    if opts.images and not rep.interrupted:
        # Only now are the indexed images (re-)embedded with this budget.
        store.set_meta("image_budget", str(opts.image_budget))
        store.delete_meta("image_budget_pending")
    store.set_meta("last_indexed_at", str(time.time()))
    return rep


def has_words(text: str, min_letters: int = MIN_PAGE_LETTERS) -> bool:
    """True when text has at least min_letters letters (digits, spaces and
    punctuation do not count; a CJK character counts as two)."""
    n = 0
    for ch in text:
        if ch.isalpha():
            n += 2 if unicodedata.east_asian_width(ch) in "WF" else 1
            if n >= min_letters:
                return True
    return False


def _mark_error(p: _Pending, ex: BaseException, prefix: str = "") -> None:
    note = f"{prefix}{type(ex).__name__}: {ex}"[:300]
    p.status, p.note, p.chunks, p.texts = "error", note, [], []


def _extract_chunks(p: _Pending, embedder: Embedder, opts: IndexOptions) -> None:
    e = p.entry
    sig = SignatureBuilder()
    if e.kind == "pdf":
        ordn = 0
        for page in read_pdf_pages(e.path):
            if not has_words(page.text):
                continue  # e.g. a figure page whose text layer is only "5"
            sig.update(page.text)
            for c in chunk_text(
                page.text, embedder.token_offsets, opts.chunk_tokens, opts.chunk_overlap
            ):
                p.chunks.append(ChunkData(ordn, None, c.text, page.number))
                p.texts.append(c.text)
                ordn += 1
    else:
        text = read_text_file(e.path)
        sig.update(text)
        for i, c in enumerate(
            chunk_text(text, embedder.token_offsets, opts.chunk_tokens, opts.chunk_overlap)
        ):
            p.chunks.append(ChunkData(i, None, c.text, None, c.line_start, c.line_end))
            p.texts.append(c.text)
    p.sig = sig.finish()
