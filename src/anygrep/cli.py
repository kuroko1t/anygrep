"""Command line interface.

`anygrep QUERY [PATH ...]` searches; the other commands are `index`,
`status`, `remove` (`rm`), `reset` and `model`. The first argument decides:
an exact command name runs that command, anything else is a search
(`search` is the internal name of the default command).
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import locale
import math
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_IMAGE_BUDGET,
    MODEL_NAME,
    MODEL_REPO,
    MODEL_REVISION,
    default_bg_threads,
    default_index_budget,
    default_index_path,
    default_threads,
    model_dir,
)
from .errors import ModelLoadError
from .extract import KINDS
from .paths import is_under as _is_under

EXIT_NOMATCH = 1
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130

TOP_HELP = """\
anygrep - search notes, PDFs and images by meaning. Local, CPU only.

usage: anygrep [OPTIONS] QUERY [PATH ...]         search (default command)
       anygrep [OPTIONS] -e QUERY [PATH ...]      QUERY starts with '-' or is a command name
       anygrep [OPTIONS] --image FILE [PATH ...]  search with an image
       anygrep [OPTIONS] -Q [PATH ...]            interactive: model stays loaded
       anygrep index [--dry-run] [PATH ...]       index or update ahead of time (no PATH: all roots)
       anygrep index --background PATH | --stop   index in a low-priority background process
       anygrep status [--problems] [--json]
       anygrep remove PATH ...                    (alias: rm) drop folders/files from the index
       anygrep reset [-y]
       anygrep model [--download]

QUERY is one argument: quote multi-word queries. Later arguments are PATHs.
PATH defaults to the current directory. New or changed files under PATH are
indexed first, with a progress bar; Ctrl-C stops and the next run continues.
When stderr is not a terminal, the new files are indexed only if that should
take about 30 s or less; otherwise none are, and the search uses the index as
it is. Home folders and drive roots are never indexed automatically.

Scope:
  -g, --glob GLOB        include paths matching GLOB, or exclude with '!GLOB' (repeatable)
  -t, --type TYPE        only text|pdf|image (repeatable or comma list)
  -T, --type-not TYPE    exclude a type
      --everywhere       search every indexed folder; no PATH, no indexing
Indexing before the search:
      --no-index         do not index or write anything; search the index as it is
  -y, --yes              index everything first, also when stderr is not a terminal
      --index-budget S   seconds of indexing allowed when stderr is not a terminal (default 30, $ANYGREP_INDEX_BUDGET)
Results:
  -m, --max-count N      total number of results (default 10; not per file as in grep)
      --weak             also show results below the relevance cutoff, marked weak
      --chunks           every matching chunk, not one per file
      --no-collapse      list near-duplicate files separately
Output:
  -l, --files-with-matches   matching file paths only, best first
  -0, --null             with -l: end each path with NUL instead of newline
      --json             one JSON object per line per result
  -q, --quiet            print nothing; exit status only
      --color WHEN       auto|always|never (default auto; NO_COLOR is honoured)
      --open             open the top result
Common:
      --db FILE          index database (default: user data dir, $ANYGREP_DB)
  -j, --threads N        CPU threads (default 8, $ANYGREP_THREADS)
  -h, --help             show this help;  anygrep COMMAND -h for a command
  -V, --version
"""

# The usage block, printed when anygrep is run without arguments.
SHORT_USAGE = "\n\n".join(TOP_HELP.split("\n\n")[:2]) + "\nRun anygrep -h for all options."

COMMANDS = frozenset({"index", "status", "remove", "rm", "reset", "model", "search", "help"})

# Auto-index (section 4 of docs/cli-spec) when stderr is not a terminal: the
# pre-scan stops after this many directory entries or seconds; the scope is
# then searched as it is. On a terminal the scan is not capped (it shows a
# count and Ctrl-C stops it).
AUTO_INDEX_MAX_ENTRIES = 20_000
AUTO_INDEX_MAX_SECONDS = 2.0
# The progress bar of an auto-index appears only after this many seconds, and
# so does the file count of the scan.
AUTO_INDEX_PROGRESS_DELAY = 1.0
SCAN_STATUS_EVERY = 0.2


class _Exit(Exception):
    """Stop the command with this exit status (the message is printed)."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


# ---------------------------------------------------------------- helpers
def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _fail(msg: str, code: int = EXIT_ERROR) -> _Exit:
    _err(msg)
    return _Exit(code)


def _use_color(stream, when: str = "auto") -> bool:
    if when == "always":
        return True
    if when == "never" or os.environ.get("NO_COLOR") or not stream.isatty():
        return False
    return sys.platform != "win32" or bool(os.environ.get("WT_SESSION"))


def _dump_json(obj, indent: int | None = 2) -> str:
    # Non-ASCII stays readable when stdout is UTF-8; otherwise escape it so
    # nothing is lost to the output encoding.
    enc = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "").replace("_", "")
    return json.dumps(obj, ensure_ascii=enc not in ("utf8", "utf8sig", "cp65001"), indent=indent)


def _clean_text(s: str) -> str:
    from .search import clean_text

    return clean_text(s)


def _decode_line(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False), "replace")


def _read_line(prompt: str) -> str:
    """input() for queries and answers, with the prompt on stderr. Piped input
    is read as UTF-8 when it decodes as such, else in the locale encoding: on
    Windows, Python decodes pipes with the ANSI code page while PowerShell may
    send UTF-8 with a BOM."""
    stdin = sys.stdin
    if stdin is None:
        raise EOFError
    sys.stderr.write(prompt)
    sys.stderr.flush()
    buf = getattr(stdin, "buffer", None)
    if buf is None or stdin.isatty():
        return _clean_text(input())
    raw = buf.readline()
    if not raw:
        raise EOFError
    return _clean_text(_decode_line(raw))


def _style(s: str, code: str, on: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if on else s


def display_path(p: str, raw: bool = False) -> str:
    """A path for terminal output: relative when under the current folder,
    control characters (possible in file names) replaced unless `raw`."""
    from .search import printable

    clean = (lambda x: x) if raw else printable
    try:
        rel = os.path.relpath(p)
    except ValueError:  # different drive on Windows
        return clean(p)
    return clean(p if rel.startswith("..") else rel)


def _tilde(p: str) -> str:
    """An absolute path with ~ for the home folder, quoted for a shell when
    needed, for messages that suggest a command."""
    from .search import printable

    with contextlib.suppress(RuntimeError, OSError):
        home = str(Path.home())
        if p == home:
            p = "~"
        elif p.startswith(home.rstrip("/\\") + os.sep):
            p = "~" + p[len(home.rstrip("/\\")) :]
    p = printable(p)
    if any(c in p for c in " \t\"'$&;|<>()*?[]{}`!#"):
        if sys.platform == "win32":
            return f'"{p}"'
        if p.startswith("~"):  # keep ~ outside the quotes so the shell expands it
            head, _, rest = p.partition(os.sep)
            return head + os.sep + shlex.quote(rest) if rest else p
        return shlex.quote(p)
    return p


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt_secs(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# " --db FILE" when the index was given with --db, for the commands that
# messages suggest (set by main()).
_DB_HINT = ""


def _hint(command: str) -> str:
    """A command to suggest, e.g. "anygrep index", naming --db if it was given."""
    return f"anygrep {command}{_DB_HINT}"


def _index_path(args) -> Path:
    path = Path(args.db).expanduser() if args.db else default_index_path()
    return path.resolve()


def _open_store(args, create: bool, readonly: bool = False):
    from .store import IndexMismatchError, Store

    path = _index_path(args)
    try:
        return Store(path, create=create, readonly=readonly)
    except FileNotFoundError:
        raise _fail(f"anygrep: no index at {path}. Run {_hint('index')} <folder> first.") from None
    except IndexMismatchError as e:
        raise _fail(f"anygrep: {e}") from None


def _make_embedder_factory(args):
    """factory(vision) -> embedder, loaded once per command (see
    embedder.embedder_factory)."""
    from .embedder import embedder_factory
    from .errors import ModelUnavailableError

    load = embedder_factory(args.threads, quiet=getattr(args, "quiet", False))

    def factory(vision: bool):
        try:
            return load(vision)
        except ModelUnavailableError as e:
            raise _fail(str(e)) from None

    return factory


def _stored_options(store, quiet: bool = True):
    """IndexOptions as the last `anygrep index` left them (extensions and image
    budget)."""
    from .indexer import IndexOptions

    pending = store.get_meta("image_budget_pending")
    budget = int(pending or store.get_meta("image_budget") or DEFAULT_IMAGE_BUDGET)
    exts = frozenset(filter(None, (store.get_meta("text_exts") or "").split(",")))
    return IndexOptions(image_budget=budget, extra_text_exts=exts, quiet=quiet)


# ---------------------------------------------------------------- commands
def cmd_index(args) -> int:
    if args.run_job:
        from .job import run_worker

        return run_worker(args)
    if args.stop:
        return _index_stop(args)
    if args.background and args.dry_run:
        raise _fail("anygrep: give either --background or --dry-run, not both")
    nothing = f"anygrep: nothing to index. Usage: {_hint('index')} <folder> [<folder> ...]"
    if args.paths:
        roots = [Path(p).expanduser() for p in args.paths]
        missing = [p for p in roots if not p.exists()]
        if missing:
            _err("anygrep: not found: " + ", ".join(map(str, missing)))
            return EXIT_ERROR
    elif not _index_path(args).exists():
        _err(nothing)
        return EXIT_ERROR
    if args.dry_run:
        return _index_dry_run(args)
    if args.background:
        return _index_background(args)

    store = _open_store(args, create=True)
    try:
        if not store.try_write_lock(probe_db=False):
            raise _busy(args, index_hint=True)
        if _job_starting(args):
            store.release_write_lock()
            raise _busy(args, index_hint=True)
        rep = _index_locked(args, store)
    finally:
        store.close()
    return _index_summary(args, rep)


def _index_locked(args, store, worker=None):
    """`anygrep index` with the writer lock held: the IndexReport, or None
    when there is nothing to index (the message is printed). `worker` is the
    background job's hooks object (job._Worker)."""
    from .indexer import BG_CALIB_KEY, CALIB_KEY, index_settings, index_targets, update_index

    nothing = f"anygrep: nothing to index. Usage: {_hint('index')} <folder> [<folder> ...]"
    budget, previous, exts = index_settings(
        store, args.image_tokens, not args.no_images, args.ext or []
    )
    if previous is not None and budget != previous and not args.no_images:
        _err(
            f"Image token budget changes from {previous} to {budget}: "
            "already indexed images will be embedded again."
        )
    if not args.paths and not store.roots():
        _err(nothing)
        return None
    roots, dropped, gone_dirs = index_targets(store, args.paths)
    if gone_dirs:
        _err("Skipping missing folders: " + ", ".join(map(str, gone_dirs)))

    opts = _index_options(args, budget, exts)
    if worker is not None:
        worker.budget = budget
        opts.quiet = True  # no progress bar in the log
        opts.should_stop = worker.should_stop
        opts.on_plan = worker.on_plan
        opts.on_progress = worker.on_progress
    calibrate = BG_CALIB_KEY if worker is not None else CALIB_KEY
    return update_index(
        store, roots, opts, _make_embedder_factory(args), dropped=dropped, calibrate=calibrate
    )


def _index_summary(args, rep) -> int:
    """Print what `anygrep index` did; the exit status."""
    if rep is None:
        return EXIT_ERROR
    for p in rep.walk.unsupported_roots:
        _err(
            f"Skipped {p}: unsupported file type"
            + (f" (use --ext {p.suffix} to index it as plain text)." if p.suffix else ".")
        )
    if args.paths:
        for root, by in rep.walk.ignored_roots:
            _err(
                f"Note: {display_path(str(root))} is excluded by the ignore files in "
                f"{display_path(str(by))}; indexing it anyway because you named it."
            )
    if rep.scanned == 0 and not rep.walk.unsupported_roots and not args.quiet:
        w = rep.walk
        _err(
            f"No files to index: {w.ignored} ignored (hidden, ignore files, dependency folders), "
            f"{w.unsupported} of unsupported types, {w.too_large} too large."
        )
    bad_dirs = rep.walk.unreadable_dirs
    if bad_dirs:
        shown = ", ".join(display_path(str(d)) for d in bad_dirs[:3])
        more = f" and {len(bad_dirs) - 3} more" if len(bad_dirs) > 3 else ""
        _err(f"Could not read {shown}{more}; files indexed there before are kept.")
    if not args.quiet:
        parts = [
            f"{rep.new} new",
            f"{rep.updated} updated",
            f"{rep.unchanged} unchanged",
            f"{rep.removed} removed",
        ]
        if rep.moved:
            parts.append(f"{rep.moved} moved")
        if rep.skipped:
            parts.append(f"{rep.skipped} skipped")
        if rep.errors:
            parts.append(f"{rep.errors} failed")
        if rep.still_failed:
            parts.append(f"{rep.still_failed} still failed from an earlier run")
        head = "Stopped after" if rep.interrupted else "Done in"
        _err(f"{head} {_fmt_secs(rep.seconds)}: " + ", ".join(parts) + ".")
        for k, n in sorted(rep.by_kind_files.items()):
            secs = rep.by_kind_seconds.get(k, 0.0)
            _err(f"  {k:5s} {n} files embedded, {secs:.1f}s ({secs / max(n, 1):.2f}s/file)")
        if rep.reused:
            n = rep.reused
            _err(f"  {n} identical image{'s' * (n != 1)} reused (not embedded again).")
        if rep.deferred_images:
            _err(f"  {rep.deferred_images} images not indexed (--no-images).")
        if rep.skipped or rep.errors or rep.still_failed:
            _err(f"  See `{_hint('status')} --problems` for skipped or failed files.")
    if rep.interrupted:
        _err("Interrupted. Finished files are saved; run the same command again to resume.")
        return EXIT_INTERRUPTED
    # A failed file that has not changed is not tried again, but it still
    # counts: a scheduled `anygrep index || alert` keeps reporting it.
    return EXIT_ERROR if rep.errors or rep.still_failed else 0


# ---------------------------------------------------------------- background
def _job_index(args) -> Path:
    return Path(os.path.abspath(_index_path(args)))


def _job_resume(st) -> str:
    return f"{_hint('index --background')} {_scope_text(st.paths)}"


def _stale_message(st) -> None:
    """A background job that died, reported once by whoever cleans it up."""
    done = f" ({st.files_done:,}/{st.files_total:,} files)" if st.files_total else ""
    _err(
        f"anygrep: the background index of {_scope_text(st.scope)} stopped before it "
        f"finished{done}; resume with: {_job_resume(st)}"
    )
    if st.error:
        _err(f"  failed: {st.error}; log: {_tilde(st.log)}")


def _job_check(args, clean: bool = True):
    """The live background job of this index or None; reports a dead one."""
    from . import job

    alive, stale = job.check(_job_index(args), clean=clean)
    if stale is not None:
        _stale_message(stale)
    return alive


def _job_meter(st) -> str:
    """The job's progress bar as one line (as `anygrep index` draws it)."""
    from .job import meter

    enc = (getattr(sys.stderr, "encoding", None) or "").lower().replace("-", "")
    line = meter(st, ascii=enc not in ("utf8", "utf8sig", "cp65001"))
    if line is None:
        return f"counting files under {_scope_text(st.scope)}"
    return line


def _job_progress(st, scope: list[str]) -> str:
    """How the background job relates to a search over `scope` (all indexed
    folders when empty): one line, without the "anygrep: " prefix."""
    from .job import overlaps

    status = f"({_hint('status')})"
    if not scope or overlaps(st.scope, scope):
        return f"background index: {_job_meter(st)} {status}"
    where = _scope_text(scope)
    if overlaps(st.paths, scope):
        return f"{where} is queued for background indexing after {_scope_text(st.scope)} {status}"
    return (
        f"a background index of {_scope_text(st.scope)} is running, so {where} was not "
        f"checked for new files; add it with: {_hint('index --background')} {where}"
    )


def _job_starting(args) -> bool:
    """With the writer lock just taken: a background job was started and
    its worker has not taken the lock yet. The caller gives the lock back
    and reports the job, as when the worker already holds it."""
    from . import job

    return job.check(_job_index(args), clean=False)[0] is not None


def _busy(args, index_hint: bool = False) -> _Exit:
    """The writer lock is taken: say by what."""
    if _job_check(args) is None:
        return _fail(BUSY)
    _err(
        f"anygrep: a background index is running ({_hint('status')}); "
        f"stop it with: {_hint('index --stop')}"
    )
    if index_hint:
        _err(f"  or add the folder to it with: {_hint('index --background')} PATH")
    return _Exit(EXIT_ERROR)


def _prefetch_model(vision: bool, quiet: bool) -> None:
    """Download missing model files in the foreground, where a network error
    is seen, so a background worker never downloads."""
    from .model import ModelUnavailableError, ensure_model

    try:
        ensure_model(vision=vision, quiet=quiet)
    except ModelUnavailableError as e:
        raise _fail(str(e)) from None


def _bg_threads(args) -> int:
    if getattr(args, "threads_given", False):
        return args.threads
    try:
        return default_bg_threads()
    except ValueError:
        return max(1, args.threads // 2)


def _request_job(args, paths: list[str], options: list[str], vision: bool):
    """Start a background job for `paths` (`anygrep index --background`) or
    add them to the running one."""
    from . import job

    index = _job_index(args)
    if job.check(index, clean=False)[0] is None:
        _prefetch_model(vision, quiet=args.quiet)
    try:
        r = job.request(index, paths, _bg_threads(args), options)
    except OSError as e:
        raise _fail(f"anygrep: could not start the background index: {e}") from None
    if r.stale is not None:
        _stale_message(r.stale)
    return r


def _index_background(args) -> int:
    """`anygrep index --background [PATH ...]`."""
    from .paths import canonical

    if args.paths:
        paths = [str(canonical(p)) for p in args.paths]
    else:
        store = _open_store(args, create=False, readonly=True)
        try:
            paths = [r for r in store.roots() if Path(r).exists()]
        finally:
            store.close()
        if not paths:
            raise _fail(
                f"anygrep: nothing to index. Usage: {_hint('index')} <folder> [<folder> ...]"
            )
    options: list[str] = []
    for e in args.ext or []:
        options += ["--ext", e]
    if args.image_tokens:
        options += ["--image-tokens", str(args.image_tokens)]
    if args.no_images:
        options.append("--no-images")
    if args.max_size:
        options += ["--max-size", repr(args.max_size)]
    if options and _job_check(args) is not None:
        raise _fail("anygrep: a background index is running; options apply only to a new job")
    r = _request_job(args, paths, options, vision=not args.no_images)
    where = _scope_text(paths)
    if r.result == "busy":
        raise _fail(BUSY)
    if r.result == "started":
        _err(
            f"anygrep: indexing {where} in the background ({_hint('status')}; "
            f"stop with {_hint('index --stop')})"
        )
    elif r.result == "cancelled":
        _err(f"anygrep: the background index of {where} was stopped before it started")
        return EXIT_INTERRUPTED
    elif r.result == "attached":
        _err(f"anygrep: {where} is already being indexed in the background ({_hint('status')})")
    else:
        _err(
            f"anygrep: {where} is queued for background indexing after "
            f"{_scope_text(r.state.scope)} ({_hint('status')})"
        )
    return 0


def _index_stop(args) -> int:
    """`anygrep index --stop`."""
    from . import job

    if args.paths or args.dry_run or args.background or args.ext or args.image_tokens:
        raise _fail("anygrep: --stop takes no PATH and no other index options")
    if args.no_images or args.max_size:
        raise _fail("anygrep: --stop takes no PATH and no other index options")
    index = _job_index(args)
    # A job just started has a state file but may not have created the
    # index yet.
    has_job = job.job_paths(index).state.exists()
    if has_job and job.check(index, clean=False)[0] is not None:
        # The worker stops after the model run in progress: say so at once.
        _err("anygrep: stopping the background index...")
    r = job.stop(index) if has_job else job.StopResult("none")
    if r.stale is not None:
        _stale_message(r.stale)
    if r.result == "none":
        print("No background index is running.")
        return 0
    st = r.state
    if r.result == "failed":
        raise _fail(f"anygrep: could not stop the background index (PID {st.pid}).")
    done = f" at {st.files_done:,}/{st.files_total:,} files" if st.files_total else ""
    print(
        f"Stopped the background index of {_scope_text(st.scope)}{done}; "
        f"resume with: {_job_resume(st)}"
    )
    return 0


def _index_options(args, budget: int, exts: set[str]):
    from .indexer import IndexOptions

    opts = IndexOptions(
        image_budget=budget,
        images=not args.no_images,
        extra_text_exts=frozenset(exts),
        quiet=args.quiet,
    )
    if args.max_size is not None:
        lim = int(args.max_size * 2**20)
        opts.max_size = {k: lim for k in KINDS}
    return opts


def _positive_mb(value: str) -> float:
    try:
        size = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--max-size must be a finite number greater than 0 MB"
        ) from None
    if not math.isfinite(size) or size <= 0 or not math.isfinite(size * 2**20):
        raise argparse.ArgumentTypeError("--max-size must be a finite number greater than 0 MB")
    return size


def _index_dry_run(args) -> int:
    """`index --dry-run`: what `index` would do, per folder (counts only, no
    time estimate); writes nothing."""
    from .indexer import IndexReport, plan, text_extensions
    from .paths import canonical
    from .store import MEMORY, Store

    path = _index_path(args)
    store = _open_store(args, create=False, readonly=True) if path.exists() else Store(MEMORY)
    try:
        budget = args.image_tokens or _stored_options(store).image_budget
        opts = _index_options(args, budget, text_extensions(store, args.ext or []))
        roots = [canonical(p) for p in args.paths] if args.paths else store.roots()
        if not roots:
            _err(f"anygrep: nothing to index. Usage: {_hint('index')} <folder> [<folder> ...]")
            return EXIT_ERROR
        for root in map(Path, roots):
            if not root.exists():
                print(f"{_tilde(str(root))}: missing, skipped")
                continue
            rep = IndexReport()
            todo, removed = plan(store, [root], opts, rep, apply=False)
            n = {k: sum(e.kind == k for e, _ in todo) for k in KINDS}
            print(
                f"{_tilde(str(root))}: {len(todo):,} new or changed files ({n['text']:,} text, "
                f"{n['pdf']:,} PDF, {n['image']:,} images), {len(removed):,} to remove"
            )
    finally:
        store.close()
    return 0


DEFAULT_N = 10
WEAK_FALLBACK = 3  # weak results shown when nothing passes the cutoff
WEAK_FALLBACK_MAX = 10  # ... or up to this many when -m is given
SIMILAR_SHOWN = 3  # paths listed in the "(+N similar: ...)" note
NO_STRONG = "No strong matches; showing the closest results."
BUSY = "anygrep: the index is being updated by another anygrep process. Try again when it is done."
NO_STRONG_PIPE = "anygrep: no strong matches (use --weak to see the closest)"
GREP_SNIPPET_COLS = 200
JSON_SNIPPET_COLS = 240


def _print_human(hits, query: str | None, color: bool, fallback: bool = False) -> None:
    """The terminal view: rank, score, location and a snippet per result."""
    from .search import snippet

    width = 160
    with contextlib.suppress(OSError):
        width = max(60, min(200, os.get_terminal_size().columns - 6))
    if not hits:
        _err("No results.")
        return
    if fallback:
        print(_style(NO_STRONG, "33", color))
    for i, h in enumerate(hits, 1):
        loc = display_path(h.path)
        if h.kind == "pdf" and h.page:
            loc += f"  p.{h.page}"
        elif h.kind == "text" and h.line_start:
            loc += f":{h.line_start}"
        tag = "" if h.kind == "text" else f"  [{h.kind}]"
        weak = _style("  (weak)", "33", color) if h.weak else ""
        print(f"{i:>2}  {h.score:.3f}  {_style(loc, '1', color)}{_style(tag, '2', color)}{weak}")
        snip = snippet(h.text, query, width)
        if snip:
            print("      " + _style(snip, "2", color))
        if h.similar:
            names = [display_path(x.path) for x in h.similar[:SIMILAR_SHOWN]]
            more = len(h.similar) - len(names)
            note = f"(+{len(h.similar)} similar: {', '.join(names)}{', ...' if more else ''})"
            print("      " + _style(note, "2", color))


def _grep_line(h, query: str | None, color: bool) -> str:
    """PATH:LINE:SNIPPET with LINE always a positive integer (fzf, vim -q)."""
    from .search import snippet

    if h.kind == "image":
        line, snip = 1, "[image]"
    elif h.kind == "pdf":
        head = f"[p.{h.page}] " if h.page else ""
        line, snip = 1, head + snippet(h.text, query, GREP_SNIPPET_COLS - len(head))
    else:
        line, snip = max(1, h.line_start or 1), snippet(h.text, query, GREP_SNIPPET_COLS)
    path = _style(display_path(h.path), "35", color)
    sep = _style(":", "36", color)
    return f"{path}{sep}{_style(str(line), '32', color)}{sep}{snip}"


def _print_grep(hits, query: str | None, color: bool) -> None:
    for h in hits:
        print(_grep_line(h, query, color))


def _print_files(hits, null: bool, color: bool) -> None:
    """-l: each file once, in rank order. With -0 the path is written as it is
    (NUL-separated output is for file names with any character)."""
    seen: set[str] = set()
    for h in hits:
        if h.path in seen:
            continue
        seen.add(h.path)
        p = display_path(h.path, raw=null)
        print(_style(p, "35", color), end="\0" if null else "\n")


def _json_entry(h, query: str | None) -> dict:
    from .search import snippet

    return {
        "path": h.path,
        "type": h.kind,
        "line_start": h.line_start,
        "line_end": h.line_end,
        "page": h.page,
        "score": round(h.score, 4),
        "weak": h.weak,
        "snippet": snippet(h.text, query, JSON_SNIPPET_COLS) or None,
    }


def _print_jsonl(hits, query: str | None) -> None:
    for i, h in enumerate(hits, 1):
        obj = {"rank": i, **_json_entry(h, query)}
        obj["similar"] = [_json_entry(x, query) for x in h.similar]
        print(_dump_json(obj, indent=None))


def _open_file(path: str) -> None:
    """Open a result with the default application: only notes, PDFs and
    images, so a script indexed with --ext (or an HTML page) is never run."""
    import stat

    from .extract import IMAGE_EXTS, PDF_EXTS, TEXT_EXTS

    openable = (TEXT_EXTS - {".html", ".htm"}) | PDF_EXTS | IMAGE_EXTS
    if Path(path).suffix.lower() not in openable:
        _err(f"Not opening {display_path(path)}: --open only opens notes, PDFs and images.")
        return
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            _err(f"Not opening {display_path(path)}: not a regular file.")
            return
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            opener = shutil.which("xdg-open") or shutil.which("wslview")
            if opener is None:
                _err(f"Cannot open {path}: install xdg-utils (or wslu on WSL).")
                return
            subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        _err(f"Cannot open {path}: {e}")


# ---------------------------------------------------------------- search
def _resolve_search_args(args) -> None:
    """Normalise the positionals (QUERY and PATHs), check option combinations
    and PATHs, and set args.query, args.paths, args.scope, args.kinds and
    args.index_budget. Raises _Exit(2) after printing the problem. Nothing
    here opens the index or loads the model."""
    from .paths import search_scope

    def bad(msg: str) -> _Exit:
        return _fail(f"anygrep: {msg}")

    plain_query = args.e is None and args.image is None and not args.interactive
    positionals = ([args.query] if args.query is not None else []) + list(args.paths)
    if plain_query:
        args.query = positionals[0] if positionals else None
        args.paths = positionals[1:]
    else:
        args.query = args.e
        args.paths = positionals
    if args.e is not None and args.image is not None:
        raise bad("give either a text query or --image, not both")
    if args.interactive:
        if args.e is not None or args.image is not None:
            raise bad("-Q reads the queries from the prompt; drop -e/--image")
        used = [
            f for f, on in (("--json", args.json), ("-l", args.files), ("-q", args.quiet)) if on
        ] + (["--open"] if args.open else [])
        if used:
            raise bad(f"{', '.join(used)}: not available with -Q")
    elif args.query is None and args.image is None:
        raise bad("give a query, -e QUERY, --image FILE or -Q")
    elif args.query is not None and not _clean_text(args.query).strip():
        # e.g. anygrep "$Q" with Q unset: a usage error, not "no matches"
        raise bad("the query is empty; give a query, -e QUERY, --image FILE or -Q")
    if args.null and not args.files:
        raise bad("-0 needs -l")
    if args.json and args.files:
        raise bad("--json cannot be combined with -l")
    if args.max_count is not None and args.max_count < 1:
        raise bad("-m needs a number of results of at least 1")
    if args.index_budget is None:
        try:
            args.index_budget = default_index_budget()
        except ValueError:
            raise bad(
                f"ANYGREP_INDEX_BUDGET={os.environ.get('ANYGREP_INDEX_BUDGET')!r} is not a "
                "number of seconds"
            ) from None
    elif not args.index_budget >= 0:
        raise bad("--index-budget must be 0 or more seconds")
    if args.everywhere:
        if args.paths:
            raise bad("--everywhere searches all indexed folders; drop PATH or drop --everywhere")
        args.no_index = True
    kinds = set(args.type or KINDS) - set(args.type_not or [])
    if not kinds:
        raise bad("-t/-T leave no file type to search")
    args.kinds = sorted(kinds) if kinds != set(KINDS) else None

    for p in args.paths:
        if p == "-":
            raise bad("reading from stdin is not supported")
        if not os.path.exists(os.path.expanduser(p)):
            _err(f"anygrep: no such file or directory: {p!r}")
            if args.line_number and p.isdigit():  # anygrep "q" -n 5
                _err(
                    "anygrep: -n takes no number (line numbers are always shown); "
                    f"for the number of results use -m {p}"
                )
            elif p.startswith("-"):  # only after `--` can a PATH start with '-'
                _err(
                    "anygrep: everything after -- is the query and PATHs; put options "
                    "before --, or use -e QUERY instead (e.g. anygrep -e status -m 2)"
                )
            elif plain_query and args.query is not None:
                _err(
                    'anygrep: did you forget to quote the query?  e.g. anygrep "how many vacation days"'
                )
            raise _Exit(EXIT_ERROR)
    args.scope = [] if args.everywhere else search_scope(args.paths or [os.getcwd()])


def _scope_text(scope: list[str]) -> str:
    return " ".join(_tilde(p) for p in scope)


def cmd_search(args) -> int:
    from .paths import is_unwalkable

    _resolve_search_args(args)
    args.qimage = _load_query_image(args.image) if args.image else None
    path = _index_path(args)
    if not path.exists():
        if args.everywhere:
            raise _fail(f"anygrep: the index is empty. Run {_hint('index')} <folder> first.")
        if args.no_index:
            raise _fail(
                f"anygrep: no index at {path}; drop --no-index to index "
                f"{_scope_text(args.scope)} first, or run {_hint('index')} <folder>."
            )
        if all(is_unwalkable(p) for p in args.scope):
            _no_indexed_files(args)
    store = _open_store(args, create=not args.no_index, readonly=args.no_index)
    try:
        return _search(args, store)
    finally:
        # An open handle would keep the index file from being deleted on Windows.
        store.close()


def _load_query_image(name: str):
    """The --image query, read before the index is opened or the model loaded."""
    from .embedder import load_image

    p = Path(name).expanduser()
    if not p.is_file():
        raise _fail(f"anygrep: no such image: {p}")
    try:
        return load_image(p)
    except (OSError, ValueError, MemoryError) as e:  # UnidentifiedImageError is an OSError
        raise _fail(f"anygrep: cannot read image {p}: {e}") from None


def _update_hint(store, scope: str) -> str:
    """The command that updates the indexed folders under `scope`."""
    roots = store.roots()
    under = [r for r in roots if _is_under(r, scope)]
    if under and len(under) == len(roots):
        return _hint("index")  # every indexed folder
    if 0 < len(under) <= 2:
        return f"{_hint('index')} {_scope_text(under)}"
    return f"{_hint('index')} <folder>"


def _no_indexed_files(args, store=None):
    from .paths import is_unwalkable

    if args.everywhere:
        raise _fail(f"anygrep: the index is empty. Run {_hint('index')} <folder> first.")
    where = _scope_text(args.scope)
    if all(is_unwalkable(p) for p in args.scope):
        # e.g. `anygrep QUERY` in a new terminal, which opens in the home folder
        _err(
            f"anygrep: nothing is indexed under {where}, and home folders and drive roots "
            "are too large to index automatically."
        )
        raise _fail(f"  index a folder in it instead: {_hint('index')} <folder>")
    _err(f"anygrep: nothing indexed under {where}")
    everywhere = store is not None and store.has_chunks()
    raise _fail(
        f"  index a folder with: {_hint('index')} <folder>"
        + ("   or search with --everywhere" if everywhere else "")
    )


def _auto_index(args, store, factory) -> int | None:
    """Section 4 of the CLI spec: index new and changed files under the
    search scope first. When stderr is a terminal (progress is visible) all of
    them are indexed, with a progress bar; otherwise only when that fits the
    time budget (all or nothing). Returns an exit status when the search must
    stop. Notes meant for after the results go to args.after_notes."""
    from .paths import is_unwalkable

    def note(msg: str) -> None:
        if not args.quiet:
            _err(msg)

    walkable = []
    for p in args.scope:
        if is_unwalkable(p):
            if not store.count_files_under([p]):
                continue  # nothing to search there either: said after the walk
            args.after_notes.append(
                f"anygrep: {_tilde(p)} is not checked for new files (home folders and drive "
                f"roots are never indexed automatically); searched the indexed part. "
                f"Update with: {_update_hint(store, p)}"
            )
        else:
            walkable.append(p)
    if not walkable:
        return None
    if args.job is not None:
        return None  # it holds the writer lock; its line comes after the results
    if not store.try_write_lock():
        note(
            "anygrep: the index is being updated by another anygrep process; "
            "searching the current snapshot."
        )
        return None
    try:
        return _auto_index_locked(args, store, factory, walkable, note)
    finally:
        store.release_write_lock()


class _ScanStatus:
    """The walk's `should_stop` hook on a terminal: after a second, one row
    with the number of files found so far, wiped when the scan ends."""

    def __init__(self, where: str, stats) -> None:
        self.where, self.stats = where, stats
        self.start = self.last = time.monotonic()
        self.shown = 0

    def __call__(self) -> bool:
        now = time.monotonic()
        if now - self.start >= AUTO_INDEX_PROGRESS_DELAY and now - self.last >= SCAN_STATUS_EVERY:
            self.last = now
            # The count first: a long path is cut, the number never is.
            line = f"anygrep: scanning {self.stats.seen:,} files in {self.where}"
            line = line[: max(20, shutil.get_terminal_size().columns - 1)]
            sys.stderr.write("\r" + line.ljust(self.shown) + "\r")
            sys.stderr.flush()
            self.shown = len(line)
        return False

    def clear(self) -> None:
        if self.shown:
            sys.stderr.write("\r" + " " * self.shown + "\r")
            sys.stderr.flush()


def _auto_index_locked(args, store, factory, walkable: list[str], note) -> int | None:
    from .extract import kind_of
    from .indexer import (
        IndexReport,
        estimate_cost,
        plan,
        record_generation,
        run_index,
        throughput_factor,
    )
    from .walker import WalkTruncated

    # Progress is visible: index the whole scope (Ctrl-C keeps what is done).
    visible = sys.stderr is not None and sys.stderr.isatty() and not args.quiet
    opts = _stored_options(store, quiet=not visible)
    opts.progress_delay = AUTO_INDEX_PROGRESS_DELAY
    roots = [Path(p) for p in walkable]
    report = IndexReport()
    where = _scope_text(walkable)
    indexed = store.count_files_under(args.scope)
    if visible:
        scan = _ScanStatus(where, report.walk)
        try:
            todo, removed = plan(store, roots, opts, report, should_stop=scan)
        finally:
            scan.clear()
    else:
        try:
            todo, removed = plan(
                store, roots, opts, report, max_entries=AUTO_INDEX_MAX_ENTRIES,
                max_seconds=AUTO_INDEX_MAX_SECONDS,
            )  # fmt: skip
        except WalkTruncated:
            if indexed:
                args.after_notes.append(
                    f"anygrep: {where} is too large to check for new files; searched the "
                    f"indexed part. Update with: {_hint('index')} {where}"
                )
                return None
            _err(f"anygrep: {where} is not indexed yet and is too large to index automatically.")
            _err(f"  index it once with: {_hint('index')} {where}")
            return EXIT_ERROR
    if removed:  # cheap updates: deleted files go (touches and moves are done)
        store.delete_files(removed)
    if not todo:
        return None

    # The estimate decides only whether a run without a visible progress bar
    # fits the budget; it is never shown.
    est = sum(estimate_cost(e, opts.image_budget) for e, _ in todo) * throughput_factor(store)
    if not (visible or args.yes or est <= args.index_budget):
        if indexed:
            args.after_notes.append(
                f"anygrep: {where} has new files not indexed yet; searched the indexed part. "
                f"Update with: {_hint('index')} {where}"
            )
            return None
        _err(
            f"anygrep: {where} is not indexed yet and is too large to index automatically "
            f"({len(todo):,} files)."
        )
        _err(f"  index it once with: {_hint('index')} {where}  (or add -y)")
        return EXIT_ERROR

    # Embed everything pending. A PATH outside the indexed folders becomes one.
    known = store.roots()
    new_roots = [
        p
        for p in walkable
        if not any(_is_under(p, r) for r in known)
        and (Path(p).is_dir() or kind_of(Path(p), opts.extra_text_exts) is not None)
    ]
    store.add_roots(new_roots)
    if store.get_meta("image_budget") is None:
        store.set_meta("image_budget", str(opts.image_budget))
    try:
        rep = run_index(store, roots, opts, factory, planned=(todo, [], report), add_roots=False)
    except KeyboardInterrupt:
        _note_added(new_roots, store, note)
        raise
    store.set_meta("last_indexed_at", str(time.time()))
    if not rep.interrupted:
        record_generation(store)
    # Also after an interrupt: the folder stays indexed and the next run resumes it.
    _note_added(new_roots, store, note)
    if rep.interrupted:
        _err("Interrupted. Finished files are saved; the next search or index continues.")
        return EXIT_INTERRUPTED
    if rep.errors or rep.skipped:
        note(
            f"anygrep: {rep.errors + rep.skipped:,} files could not be indexed; "
            f"see {_hint('status')} --problems"
        )
    return None


def _nothing_indexed_with_job(args) -> None:
    """Exit 2: nothing in the scope is searchable yet and a background job
    holds the index (so the scope cannot be indexed now)."""
    from .job import overlaps
    from .paths import is_unwalkable

    st, scope = args.job, args.scope
    if all(is_unwalkable(p) for p in scope):
        return  # the home folder message says what to do
    where = _scope_text(scope)
    if overlaps(st.paths, scope):
        raise _fail(f"anygrep: nothing under {where} is indexed yet; {_job_progress(st, scope)}")
    _err(
        f"anygrep: nothing under {where} is indexed yet; a background index of "
        f"{_scope_text(st.scope)} is running."
    )
    raise _fail(f"  add it to the job with: {_hint('index --background')} {where}")


def _job_notes(args) -> list[str]:
    """The line about a background job that runs while this search does."""
    if args.job is None:
        return []
    from . import job

    st = job.read_state(_job_index(args))
    if st is None or job.liveness(_job_index(args), st) == "stale":
        return []
    return [f"anygrep: {_job_progress(st, args.scope)}"]


def _outdated_notes(args, store) -> list[str]:
    """The line about files that an older anygrep processed (see
    config.PROCESSING_GENERATION) and that this search did not index again."""
    n = store.count_stale(None if args.everywhere else args.scope)
    if not n:
        return []
    where = "" if args.everywhere else f" {_scope_text(args.scope)}"
    files = f"{n:,} file{'s' if n != 1 else ''}"
    return [
        f"anygrep: {files} here {'were' if n != 1 else 'was'} indexed by an older version of "
        f"anygrep and will be indexed again by the next update; their results may differ "
        f"until then. Update now with: {_hint('index')}{where}"
    ]


def _note_added(new_roots: list[str], store, note) -> None:
    for p in new_roots:
        n = store.count_files_under([p])
        note(
            f"anygrep: added {_tilde(p)} as an indexed folder ({n:,} files). "
            f"Undo: {_hint('rm')} {_tilde(p)}"
        )


def _search(args, store) -> int:
    from .paths import canonical
    from .search import image_query, query_budget, scope_matcher, search

    args.after_notes = []
    qimage = args.qimage
    factory = _make_embedder_factory(args)
    # A dead background job is cleaned up (and reported) unless nothing may
    # be written; -q leaves the report to the next command.
    args.job = _job_check(args, clean=not (args.no_index or args.quiet))
    if not args.no_index:
        code = _auto_index(args, store, factory)
        if code is not None:
            return code
    if args.everywhere:
        if not store.has_chunks():
            _no_indexed_files(args)
    elif store.count_files_under(args.scope) == 0:
        if args.job is not None:
            _nothing_indexed_with_job(args)
        _no_indexed_files(args, store)
    args.after_notes.extend(_job_notes(args))
    args.after_notes.extend(_outdated_notes(args, store))

    budget = query_budget(store)
    embedder = factory(bool(args.image))
    embedder.image_budget = budget
    scope = scope_matcher(args.scope, args.glob or [])
    n = DEFAULT_N if args.max_count is None else args.max_count
    if args.interactive or (sys.stdout.isatty() and not (args.files or args.json)):
        mode = "human"
    else:
        mode = "json" if args.json else "files" if args.files else "grep"
    if args.quiet:
        mode = "quiet"
    # Output meant for other programs is coloured only with --color always.
    if mode in ("files", "grep"):
        color = args.color == "always"
    else:
        color = _use_color(sys.stdout, args.color) and mode != "json"

    # Output for other programs (grep format, -l) lists every file, like grep:
    # it has no room for a "(+N similar)" note, so collapsed files would be lost.
    collapse = not args.no_collapse and mode not in ("grep", "files")

    def run(
        qvec, query_text: str | None, exclude: str | None = None, image_queries: dict | None = None
    ) -> list:
        def find(n: int, hide_weak: bool) -> list:
            return search(
                store, qvec, n=n, kinds=args.kinds, scope=scope,
                per_file=not args.chunks, exclude_path=exclude,
                query_kind="text" if query_text is not None else "image",
                budget=budget, hide_weak=hide_weak, collapse=collapse,
                image_queries=image_queries, prefixes=args.scope,
            )  # fmt: skip

        hits = find(n, hide_weak=not args.weak)
        if mode == "human":
            fallback = False
            if not hits and not args.weak:
                # Nothing passes the cutoff: show the closest few, marked as weak.
                few = WEAK_FALLBACK if args.max_count is None else min(n, WEAK_FALLBACK_MAX)
                hits = find(few, hide_weak=False)
                fallback = bool(hits)
            _print_human(hits, query_text, color, fallback)
        elif mode != "quiet":
            if not hits:
                _err(NO_STRONG_PIPE)
            elif mode == "json":
                _print_jsonl(hits, query_text)
            elif mode == "files":
                _print_files(hits, args.null, color)
            else:
                _print_grep(hits, query_text, color)
        return [h for h in hits if args.weak or not h.weak]

    if args.interactive:
        for msg in args.after_notes:  # there are no results yet to print them after
            _err(msg)
        eof = "Ctrl-Z Enter" if sys.platform == "win32" else "Ctrl-D"
        _err(f"anygrep: type a query; an empty line or {eof} quits.")
        while True:
            try:
                q = _read_line("> ").strip()
            except (EOFError, KeyboardInterrupt):
                _err("")
                break
            if not q:
                break
            run(embedder.embed_query(q), q)
        return 0

    if qimage is not None:
        excluded = str(canonical(args.image))
        try:
            qvec, image_queries = image_query(
                store, embedder, qimage, budget, args.kinds, args.scope, scope, excluded
            )
        except ValueError as e:  # e.g. too small for the vision model
            raise _fail(f"anygrep: cannot use image {args.image}: {e}") from None
        # Stored paths are canonical, so compare in that form.
        matched = run(qvec, None, exclude=excluded, image_queries=image_queries)
    else:
        q = _clean_text(args.query)
        matched = run(embedder.embed_query(q), q)
    if not args.quiet:
        for msg in args.after_notes:
            _err(msg)
    if args.open and matched and not args.quiet:
        _open_file(matched[0].path)
    return 0 if matched else EXIT_NOMATCH


def cmd_status(args) -> int:
    from .indexer import CALIB_KEY, throughput_factor
    from .model import missing_files
    from .search import printable

    path = _index_path(args)
    exists = path.exists()
    info: dict = {
        "index": str(path),
        "exists": exists,
        "model": MODEL_NAME,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_dir": str(model_dir()),
        "model_files_missing": [f.path for f in missing_files(vision=True)],
        "threads": args.threads,
    }
    if exists:
        store = _open_store(args, create=False)
        size = sum(
            (path.with_name(path.name + s)).stat().st_size
            for s in ("", "-wal", "-shm")
            if path.with_name(path.name + s).exists()
        )
        info.update(
            size_bytes=size,
            image_budget=int(store.get_meta("image_budget") or DEFAULT_IMAGE_BUDGET),
            image_budget_pending=int(store.get_meta("image_budget_pending") or 0) or None,
            image_budgets_indexed={str(k): v for k, v in sorted(store.image_budgets().items())},
            extra_text_exts=[e for e in (store.get_meta("text_exts") or "").split(",") if e],
            roots=store.roots(),
            counts=store.counts(),
            last_indexed_at=float(store.get_meta("last_indexed_at") or 0) or None,
            throughput_factor=(
                throughput_factor(store) if store.get_meta(CALIB_KEY) is not None else None
            ),
            outdated_files=store.count_stale(),
        )
        if args.problems:
            info["problems"] = [
                {"path": p, "status": s, "note": n} for p, s, n in store.problem_files(200)
            ]
        store.close()
    job = _job_check(args) if exists else None
    if job is not None:
        from dataclasses import asdict

        from .job import eta_seconds

        info["job"] = {**asdict(job), "eta_seconds": eta_seconds(job), "alive": True}
    else:
        info["job"] = None
    if args.json:
        print(_dump_json(info))
        return 0
    print(f"anygrep {__version__}")
    print(f"Index:   {info['index']}" + ("" if exists else "  (not created yet)"))
    print(f"Model:   {MODEL_NAME}, revision {MODEL_REVISION[:12]}")
    miss = info["model_files_missing"]
    print(f"Files:   {info['model_dir']}" + (f"  ({len(miss)} missing)" if miss else ""))
    print(f"Threads: {args.threads}")
    if not exists:
        return 0
    print(f"Size:    {_human_bytes(info['size_bytes'])}")
    mixed = info["image_budgets_indexed"]
    notes = []
    if info["image_budget_pending"]:
        notes.append(f"changing to {info['image_budget_pending']}")
    if len(mixed) > 1:
        notes.append(
            "indexed images are mixed: " + ", ".join(f"{n} at {b}" for b, n in mixed.items())
        )
    if notes:
        notes[-1] += f"; run `{_hint('index')}` to finish"
    print(
        f"Image budget: {info['image_budget']} tokens per image"
        + (f"  ({'; '.join(notes)})" if notes else "")
    )
    if info["throughput_factor"] is not None:
        print(f"Indexing: measured speed: {info['throughput_factor']:.1f}x the default estimate")
    if info["extra_text_exts"]:
        print("Extra text extensions: " + " ".join(info["extra_text_exts"]))
    if info["last_indexed_at"]:
        print(
            "Last indexed: "
            + time.strftime("%Y-%m-%d %H:%M", time.localtime(info["last_indexed_at"]))
        )
    if info["outdated_files"]:
        print(
            f"Outdated: {info['outdated_files']:,} files indexed by an older version of anygrep; "
            f"run `{_hint('index')}` to index them again"
        )
    if job is not None:
        _print_job(job)
    print("Indexed paths:")
    for r in info["roots"] or ["(none)"]:
        print(f"  {r}")
    print(f"{'type':6s} {'files':>7s} {'chunks':>8s} {'skipped':>8s} {'failed':>7s}")
    for k in KINDS:
        c = info["counts"].get(k, {})
        print(
            f"{k:6s} {c.get('ok', 0):7d} {c.get('chunks', 0):8d} "
            f"{c.get('skipped', 0):8d} {c.get('error', 0):7d}"
        )
    if args.problems:
        for p in info["problems"]:
            print(f"  {p['status']:7s} {display_path(p['path'])}: {printable(p['note'] or '')}")
    return 0


def _print_job(st) -> None:
    """The status block of a running background job."""
    started = time.localtime(st.started_at)
    fmt = "%H:%M" if started[:3] == time.localtime()[:3] else "%Y-%m-%d %H:%M"
    print(
        f"Background: indexing {_scope_text(st.scope)}, PID {st.pid}, started "
        f"{time.strftime(fmt, started)}, {st.threads} thread{'s' if st.threads != 1 else ''}"
    )
    if st.phase == "starting":
        print("  starting")
    elif st.phase == "planning":
        print("  counting files")
    else:
        if st.phase == "text":
            order = "text and PDF first, then images newest first"
        elif st.text_total:
            order = "text and PDF done, images newest first"
        else:
            order = "images newest first"
        print(f"  {_job_meter(st)}  ({order})")
    if st.queue:
        print("  then: " + "; ".join(_scope_text(item) for item in st.queue))
    print(f"  log: {_tilde(st.log)}   stop: {_hint('index --stop')}")


def cmd_remove(args) -> int:
    from .paths import canonical

    store = _open_store(args, create=False)
    if not store.try_write_lock(probe_db=False) or _job_starting(args):
        store.close()
        raise _busy(args)
    total = 0
    missing: list[str] = []
    try:
        roots = store.roots()
        for p in args.paths:
            found = False
            for ap in dict.fromkeys([str(canonical(p)), os.path.abspath(os.path.expanduser(p))]):
                ids = [r.id for r in store.files_under(ap)]
                prefix = ap.rstrip("/\\") + os.sep
                below = [r for r in roots if r == ap or r.startswith(prefix)]
                found = found or bool(ids or below)
                total += store.delete_files(ids)
                store.remove_roots(below)  # with any indexed folders inside it
                roots = [r for r in roots if r not in below]
                parent = next((r for r in roots if ap.startswith(r.rstrip("/\\") + os.sep)), None)
                if ids and parent is not None:
                    _err(
                        f"Note: {display_path(ap)} is inside the indexed folder "
                        f"{display_path(parent)}, so the next `{_hint('index')}` adds it again. "
                        "To keep it out, add it to "
                        f"{display_path(os.path.join(parent, '.anygrepignore'))}."
                    )
            if not found:
                missing.append(p)
    finally:
        store.close()
    print(f"Removed {total} files from the index.")
    for p in missing:
        _err(f"anygrep: not in the index: {p}")
    return EXIT_ERROR if missing else 0


def cmd_reset(args) -> int:
    path = _index_path(args)
    if not path.exists():
        print(f"No index at {path}.")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            _err("anygrep: refusing to delete without --yes (stdin is not a terminal).")
            return EXIT_ERROR
        try:
            ans = input(f"Delete the index at {path}? Model files are kept. [y/N] ")
        except EOFError:  # e.g. stdin is NUL on Windows, which counts as a terminal
            print()
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return EXIT_ERROR
    from .store import acquire_writer_lock

    lock = acquire_writer_lock(path)
    if lock is not None and _job_starting(args):
        lock.close()
        lock = None
    if lock is None:
        raise _busy(args)
    try:
        for s in ("", "-wal", "-shm"):
            path.with_name(path.name + s).unlink(missing_ok=True)
    except PermissionError as e:
        # Windows does not delete files another process has open.
        _err(
            f"anygrep: cannot delete {path}: {e.strerror}. "
            "Close other anygrep processes and try again."
        )
        return EXIT_ERROR
    finally:
        lock.close()
    print(f"Deleted {path}.")
    return 0


def cmd_model(args) -> int:
    from .model import ModelUnavailableError, ensure_model, missing_files, needed_files

    if args.download:
        try:
            d = ensure_model(vision=True)
        except ModelUnavailableError as e:
            _err(str(e))
            return EXIT_ERROR
        print(f"Model files ready in {d}")
        return 0
    miss = {f.path for f in missing_files(vision=True)}
    print(f"{MODEL_NAME}\n{MODEL_REPO} @ {MODEL_REVISION}\n{model_dir()}")
    files = needed_files(vision=True)
    for f in files:
        state = "missing" if f.path in miss else "ok"
        print(f"  {state:7s} {f.path} ({_human_bytes(f.size)})")
    print(f"  total   {_human_bytes(sum(f.size for f in files))}")
    if miss:
        print("Run `anygrep model --download` to fetch them now (otherwise on first use).")
    return 0


# ---------------------------------------------------------------- parser
class _Parser(argparse.ArgumentParser):
    """argparse with `anygrep: ` messages and exit status 2."""

    def __init__(self, *a, **kw) -> None:
        kw.setdefault("allow_abbrev", False)
        super().__init__(*a, **kw)

    def error(self, message: str):  # type: ignore[override]
        _err(f"anygrep: {message}")
        _err(f"anygrep: see {self.prog} -h")
        raise _Exit(EXIT_ERROR)


class _TopHelp(argparse.Action):
    def __init__(
        self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None
    ):
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        sys.stdout.write(TOP_HELP)
        raise _Exit(0)


def _kinds(value: str) -> list[str]:
    """-t/-T value: one type or a comma list."""
    out = [v.strip().lower() for v in value.split(",") if v.strip()]
    bad = [v for v in out if v not in KINDS]
    if bad or not out:
        raise argparse.ArgumentTypeError(
            f"unknown type {', '.join(bad) or repr(value)} (use {'|'.join(KINDS)})"
        )
    return out


def _common() -> argparse.ArgumentParser:
    common = _Parser(add_help=False)
    common.add_argument(
        "--db", metavar="FILE", help="index database (default: user data dir, $ANYGREP_DB)"
    )
    common.add_argument(
        "-j",
        "--threads",
        type=int,
        metavar="N",
        help="CPU threads for the model (default: half the CPU threads up to 8, or $ANYGREP_THREADS)",
    )
    return common


def build_search_parser(common=None) -> argparse.ArgumentParser:
    ps = _Parser(prog="anygrep", parents=[common or _common()], add_help=False)
    ps.add_argument("-h", "--help", action=_TopHelp)
    ps.add_argument("query", nargs="?")
    ps.add_argument("paths", nargs="*")
    ps.add_argument("-e", metavar="QUERY")
    ps.add_argument("--image", metavar="FILE")
    ps.add_argument("-Q", "--interactive", action="store_true")
    ps.add_argument("-g", "--glob", action="append", metavar="GLOB")
    ps.add_argument("-t", "--type", type=_kinds, action="extend", metavar="TYPE")
    ps.add_argument("-T", "--type-not", type=_kinds, action="extend", metavar="TYPE")
    ps.add_argument("--everywhere", action="store_true")
    ps.add_argument("--no-index", action="store_true")
    ps.add_argument("-y", "--yes", action="store_true")
    ps.add_argument("--index-budget", type=float, metavar="S")
    ps.add_argument("-m", "--max-count", type=int, metavar="N")
    ps.add_argument("--weak", action="store_true")
    ps.add_argument("--chunks", action="store_true")
    ps.add_argument("--no-collapse", action="store_true")
    ps.add_argument("-l", "--files-with-matches", dest="files", action="store_true")
    ps.add_argument("-0", "--null", action="store_true")
    ps.add_argument("--json", action="store_true")
    ps.add_argument("-q", "--quiet", action="store_true")
    ps.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    ps.add_argument("--open", action="store_true")
    # grep habits: line numbers are always shown and matching ignores case.
    ps.add_argument("-n", "--line-number", action="store_true", help=argparse.SUPPRESS)
    ps.add_argument("-i", "--ignore-case", action="store_true", help=argparse.SUPPRESS)
    ps.set_defaults(func=cmd_search)
    return ps


def build_parsers() -> dict[str, argparse.ArgumentParser]:
    """One parser per command (the command name is argv[0])."""
    common = _common()

    def sub(name: str, description: str) -> argparse.ArgumentParser:
        return _Parser(prog=f"anygrep {name}", parents=[common], description=description)

    pi = sub("index", "Index folders or files (incremental; no PATH: update all indexed folders).")
    pi.add_argument("paths", nargs="*", metavar="PATH", help="folders or files to index")
    pi.add_argument(
        "--dry-run",
        action="store_true",
        help="show how many files would be indexed or removed; change nothing",
    )
    pi.add_argument(
        "--image-tokens",
        type=int,
        choices=(70, 140, 280),
        help="tokens per image: 70 fast, 140 default, 280 for text-heavy images",
    )
    pi.add_argument("--no-images", action="store_true", help="skip images in this run")
    pi.add_argument(
        "--ext",
        action="append",
        metavar="EXT",
        help="also index files with this extension as plain text (repeatable, remembered)",
    )
    pi.add_argument(
        "--max-size", type=_positive_mb, metavar="MB", help="skip files larger than this"
    )
    pi.add_argument("-q", "--quiet", action="store_true", help="no progress output")
    pi.add_argument(
        "--background",
        action="store_true",
        help="index in a detached, low-priority process; search meanwhile (no PATH: all roots)",
    )
    pi.add_argument("--stop", action="store_true", help="stop the background index")
    pi.add_argument("--run-job", action="store_true", help=argparse.SUPPRESS)
    pi.add_argument("--job-id", help=argparse.SUPPRESS)
    pi.set_defaults(func=cmd_index)

    pst = sub("status", "Show index and model information.")
    pst.add_argument("--json", action="store_true")
    pst.add_argument("--problems", action="store_true", help="list skipped and failed files")
    pst.set_defaults(func=cmd_status)

    pr = sub("remove", "Remove folders or files from the index (alias: rm).")
    pr.add_argument("paths", nargs="+", metavar="PATH")
    pr.set_defaults(func=cmd_remove)

    prs = sub("reset", "Delete the whole index (model files are kept).")
    prs.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    prs.set_defaults(func=cmd_reset)

    pm = sub("model", "Show or download the model files.")
    pm.add_argument("--download", action="store_true", help="download all model files now")
    pm.set_defaults(func=cmd_model)
    return {
        "search": build_search_parser(common),
        "index": pi,
        "status": pst,
        "remove": pr,
        "reset": prs,
        "model": pm,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse argv after the argv shim: argv[0] is a command name."""
    cmd, rest = argv[0], argv[1:]
    parser = build_parsers()[cmd]
    if cmd != "search":
        return parser.parse_args(rest)
    rest, values, after = _split_search_args(rest)
    bad_n = next((a for a in rest if re.fullmatch(r"-n\d+", a)), None)
    if bad_n:  # grep's -n takes no number either; a count was probably meant
        parser.error(
            f"{bad_n}: -n takes no number (line numbers are always shown); "
            f"for the number of results use -m {bad_n[2:]}"
        )
    # QUERY and PATHs may be mixed with options: anygrep "q" -m 3 ~/notes
    args = parser.parse_intermixed_args(rest)
    for dest, value in values.items():
        setattr(args, dest, value)
    if after:  # everything after -- is QUERY and PATHs
        if args.query is None:
            args.query, after = after[0], after[1:]
        args.paths = [*args.paths, *after]
    return args


# Search options that take a value, and those whose value may start with '-'.
_VALUE_OPTIONS = frozenset(
    {"-g", "--glob", "-t", "--type", "-T", "--type-not", "--index-budget", "-m",
     "--max-count", "--color", "--db", "-j", "--threads"}
)  # fmt: skip
_ANY_VALUE_OPTIONS = {"-e": "e", "--image": "image"}


def _split_search_args(args: list[str]) -> tuple[list[str], dict[str, str], list[str]]:
    """Take out the values of -e and --image, which may start with '-'
    (`anygrep -e -foo`, `-e -5`), and the arguments after `--`; argparse
    would read the former as options and, before Python 3.12, mishandles
    the latter. Returns (the other arguments, {dest: value}, after --)."""
    out: list[str] = []
    values: dict[str, str] = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            return out, values, args[i + 1 :]
        if a in _ANY_VALUE_OPTIONS and i + 1 < len(args):
            values[_ANY_VALUE_OPTIONS[a]] = args[i + 1]
            i += 2
            continue
        out.append(a)
        if a in _VALUE_OPTIONS and i + 1 < len(args):
            out.append(args[i + 1])
            i += 1
        i += 1
    return out, values, []


def _setup_stdout() -> None:
    # Snippets and paths can contain any character. Pipes get UTF-8 (what JSON
    # consumers expect); terminals that cannot show a character get "?".
    out = sys.stdout
    if not hasattr(out, "reconfigure"):
        return
    with contextlib.suppress(Exception):
        if out.isatty():
            out.reconfigure(errors="replace")
        else:
            out.reconfigure(encoding=_windows_pipe_encoding() or "utf-8", errors="replace")


def _windows_pipe_encoding() -> str | None:
    """The console code page on Windows, which PowerShell and cmd tools use to
    decode a program's piped or redirected output; None elsewhere, under
    PYTHONUTF8/PYTHONIOENCODING, or without a console."""
    if sys.platform != "win32" or sys.flags.utf8_mode or os.environ.get("PYTHONIOENCODING"):
        return None
    try:
        import codecs
        import ctypes

        cp = ctypes.windll.kernel32.GetConsoleOutputCP()  # type: ignore[attr-defined]
        return codecs.lookup(f"cp{cp}").name if cp else None
    except (AttributeError, OSError, LookupError):
        return None


def _command_first(argv: list[str]) -> list[str]:
    """Move a command name that follows the global options (-j N, --db FILE)
    to the front: `anygrep -j 12 index PATH` is `anygrep index -j 12 PATH`,
    not a search for "index"."""
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-j", "--threads", "--db"):
            i += 2
        elif a.startswith(("--threads=", "--db=")) or (a.startswith("-j") and a[2:].isdigit()):
            i += 1
        else:
            break
    if 0 < i < len(argv) and argv[i] in COMMANDS:
        return [argv[i], *argv[:i], *argv[i + 1 :]]
    return argv


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _err(SHORT_USAGE)
        return EXIT_ERROR
    argv = _command_first(argv)
    first = argv[0]
    if first in COMMANDS:
        if first == "help":
            sys.stdout.write(TOP_HELP)
            return 0
        if first == "rm":
            argv[0] = "remove"
    elif first in ("-h", "--help"):
        sys.stdout.write(TOP_HELP)
        return 0
    elif first in ("-V", "--version"):
        print(f"anygrep {__version__}")
        return 0
    else:
        argv = ["search", *argv]
    args = None
    try:
        global _DB_HINT
        _DB_HINT = ""
        args = parse_args(argv)
        if getattr(args, "db", None):
            _DB_HINT = " --db " + _tilde(os.path.abspath(os.path.expanduser(args.db)))
        _setup_stdout()
        args.threads_given = args.threads is not None
        if args.threads is None:
            try:
                args.threads = default_threads()
            except ValueError:
                _err(
                    f"anygrep: ANYGREP_THREADS={os.environ.get('ANYGREP_THREADS')!r} is not a "
                    "number of threads >= 1"
                )
                return EXIT_ERROR
        elif args.threads < 1:
            _err("anygrep: --threads must be at least 1")
            return EXIT_ERROR
        code = int(args.func(args) or 0)
        # Flush here: a flush at interpreter exit would fail outside these handlers.
        sys.stdout.flush()
        return code
    except _Exit as e:
        return e.code
    except SystemExit as e:  # argparse --help of a command
        return e.code if isinstance(e.code, int) else EXIT_ERROR
    except ModelLoadError as e:
        _err(str(e))
        return EXIT_ERROR
    except sqlite3.OperationalError as e:
        msg = str(e)
        where = _index_path(args) if args is not None else "the index"
        if "locked" in msg:
            _err(
                "anygrep: the index is busy: another anygrep process is writing to it. Try again later."
            )
        elif "full" in msg:
            _err(
                f"anygrep: cannot write the index at {where}: the disk is full. "
                "Free some space and run the command again; finished work is kept."
            )
        elif "readonly" in msg:
            _err(f"anygrep: cannot write the index at {where}: it is read-only.")
        else:
            raise
        return EXIT_ERROR
    except sqlite3.DatabaseError as e:
        where = _index_path(args) if args is not None else "the index"
        _err(
            f"anygrep: cannot read the index at {where}: {e}. If it is damaged, "
            f"run `{_hint('reset')}` and index again."
        )
        return EXIT_ERROR
    except KeyboardInterrupt:
        _err("Interrupted.")
        return EXIT_INTERRUPTED
    except BrokenPipeError:
        # e.g. `anygrep ... | head`: the reader has what it wanted.
        _discard_stdout()
        return 0
    except OSError as e:
        if e.errno == errno.EINVAL and sys.platform == "win32" and not _stdout_writable():
            _discard_stdout()  # how Windows reports a closed pipe
            return 0
        if e.errno not in (errno.ENOSPC, errno.EIO):
            raise
        _discard_stdout()  # e.g. `anygrep ... > /dev/full`
        _err(f"anygrep: write error: {e.strerror}")
        return EXIT_ERROR


def _stdout_writable() -> bool:
    try:
        sys.stdout.flush()
    except OSError:
        return False
    return True


def _discard_stdout() -> None:
    """Point stdout at the null device so the flush at exit cannot fail again."""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(fd, sys.stdout.fileno())
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
