"""Directory walking with gitignore-style ignore files.

Skipped: hidden files and directories (dot-prefixed, the hidden attribute on
Windows, the hidden flag on macOS), common dependency/cache directories,
virtualenvs, anything matched by .gitignore / .ignore / .anygrepignore files
found along the way (deeper files override shallower ones, as in git),
unsupported extensions, and files over the size limit for their type.

Ignore files in the folders above a root also apply, up to the enclosing git
repository or an enclosing root that was indexed before, so indexing a
subfolder gives the same result as indexing its parent. The exception is a
root that those rules exclude (a gitignored notes folder inside a repository,
or a home folder under a dotfiles repository that ignores `*`): the user named
it, so it is indexed without the rules from above it, and the walker reports
which folder's ignore files were skipped.
"""

from __future__ import annotations

import os
import re
import stat
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

from .extract import MAX_SIZE, kind_of

IGNORE_FILES = (".gitignore", ".ignore", ".anygrepignore")
SKIP_DIRS = frozenset(
    {"node_modules", "__pycache__", "site-packages", "bower_components", "$RECYCLE.BIN"}
)


@dataclass(frozen=True)
class FileEntry:
    path: Path  # absolute
    kind: str
    size: int
    mtime_ns: int


@dataclass
class WalkStats:
    seen: int = 0
    ignored: int = 0
    too_large: int = 0
    unsupported: int = 0
    # Folders that could not be listed. Index rows below them are kept.
    unreadable_dirs: list[Path] = field(default_factory=list)
    # Roots whose stat failed with ENOENT; the indexer distinguishes directories
    # from standalone files using their existing indexed rows.
    missing_roots: list[Path] = field(default_factory=list)
    # Files only over a user-given size limit. Their index rows are kept.
    held: set[str] = field(default_factory=set)
    # Files named explicitly that have an unsupported type.
    unsupported_roots: list[Path] = field(default_factory=list)
    # (root, folder) for roots excluded by the ignore files in that folder
    # above them; they are indexed without those rules.
    ignored_roots: list[tuple[Path, Path]] = field(default_factory=list)


class WalkTruncated(Exception):
    """The walk hit its entry or time cap (a search's pre-scan)."""

    def __init__(self, entries: int, by_time: bool) -> None:
        super().__init__(f"stopped after {entries} entries")
        self.entries = entries
        self.by_time = by_time


def _is_hidden(name: str, st: os.stat_result | None) -> bool:
    if name.startswith("."):
        return True
    if st is None:
        return False
    if sys.platform == "win32":
        attrs = getattr(st, "st_file_attributes", 0)
        return bool(attrs & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))
    if sys.platform == "darwin":
        return bool(getattr(st, "st_flags", 0) & stat.UF_HIDDEN)  # e.g. ~/Library
    return False


def _is_venv(d: Path) -> bool:
    try:
        return (d / "pyvenv.cfg").exists()
    except OSError:  # e.g. a folder we may not enter
        return False


class _Rules:
    """The rules of one folder's ignore files. Most paths match none of them;
    one regex made of all the rules rejects those in a single call instead of
    trying every rule in turn."""

    def __init__(self, lines: list[str]) -> None:
        self.spec = pathspec.GitIgnoreSpec.from_lines(lines)
        # pathspec applies each rule with regex.search(). Rules anchored with
        # '^' (nearly all) are combined into one regex tried with match();
        # the others (e.g. '*/' or '**/') into one tried with search().
        self.anchored: re.Pattern[str] | None = None
        self.floating: re.Pattern[str] | None = None
        self.fast = False
        anchored, floating = [], []
        for pat in self.spec.patterns:
            if pat.include is None:  # a comment or blank line
                continue
            rx = getattr(pat, "regex", None)
            if not isinstance(getattr(rx, "pattern", None), str) or rx.flags & re.I:
                return  # unexpected: always ask pathspec
            src = re.sub(r"\(\?P<\w+>", "(?:", rx.pattern)
            (anchored if src.startswith("^") else floating).append(src)
        try:
            self.anchored = re.compile("|".join(f"(?:{x})" for x in anchored) or "(?!)")
            self.floating = re.compile("|".join(f"(?:{x})" for x in floating)) if floating else None
        except re.error:
            return
        self.fast = True

    def check(self, rel: str) -> bool | None:
        """True: ignored, False: re-included by a '!' rule, None: no rule matches."""
        if (
            self.fast
            and self.anchored.match(rel) is None  # type: ignore[union-attr]
            and (self.floating is None or self.floating.search(rel) is None)
        ):
            return None
        return self.spec.check_file(rel).include


def _rel(p: Path, base: Path) -> str:
    """p relative to base with '/' separators (Path.relative_to is slow)."""
    b, s = str(base), str(p)
    if not b.endswith(os.sep):
        b += os.sep
    if not s.startswith(b):
        return p.relative_to(base).as_posix()
    r = s[len(b) :]
    return r if os.sep == "/" else r.replace(os.sep, "/")


def _decide(p: Path, is_dir: bool, specs: list[tuple[Path, _Rules]]) -> Path | None:
    """Apply ignore specs like git: the deepest file with a matching rule
    decides. Returns the folder of the ignore file that excludes p, or None."""
    for base, rules in reversed(specs):
        rel = _rel(p, base) + ("/" if is_dir else "")
        include = rules.check(rel)
        if include is not None:
            return base if include else None
    return None


def _is_ignored(p: Path, is_dir: bool, specs: list[tuple[Path, _Rules]]) -> bool:
    return _decide(p, is_dir, specs) is not None


def _ancestor_specs(
    root: Path, stops: frozenset[Path]
) -> tuple[list[tuple[Path, _Rules]], Path | None]:
    """Ignore specs of the folders above root, outermost first, and the folder
    whose ignore file excludes root itself (then no specs are returned).

    Only folders up to the enclosing git repository root or an enclosing
    known root are used. Without either, no ancestor rules apply.
    """
    chain: list[Path] = []
    for d in root.parents:
        chain.append(d)
        try:
            if d in stops or (d / ".git").exists():
                break
        except OSError:
            return [], None
    else:
        return [], None
    out = []
    for d in reversed(chain):
        spec = _load_spec(d)
        if spec is not None:
            out.append((d, spec))
    # The folders from just below the outermost one down to root, as git would
    # walk them: an excluded folder hides everything below it.
    for f in [*reversed(chain[:-1]), root]:
        by = _decide(f, True, [(b, sp) for b, sp in out if b in f.parents])
        if by is not None:
            return [], by
    return out, None


# Larger ignore files are read only up to this size.
MAX_IGNORE_FILE_BYTES = 1 << 20


def _read_ignore_file(p: Path) -> str | None:
    """The text of a regular ignore file. Symlinks, FIFOs and devices are
    treated as absent: reading them could block or never end."""
    try:
        st = p.lstat()
        if not stat.S_ISREG(st.st_mode):
            return None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(p, flags)
        with os.fdopen(fd, "rb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):  # replaced since lstat
                return None
            data = f.read(MAX_IGNORE_FILE_BYTES)
    except OSError:
        return None
    return data.decode("utf-8", errors="replace")


def _load_spec(d: Path) -> _Rules | None:
    lines: list[str] = []
    for name in IGNORE_FILES:
        text = _read_ignore_file(d / name)
        if text is not None:
            lines += text.splitlines()
    if not lines:
        return None
    # The same ignore file often appears in many folders (vendored or
    # generated trees): build its rules once.
    key = "\n".join(lines)
    rules = _RULES_CACHE.get(key)
    if rules is None:
        if len(_RULES_CACHE) >= 256:
            _RULES_CACHE.clear()
        rules = _RULES_CACHE[key] = _Rules(lines)
    return rules


_RULES_CACHE: dict[str, _Rules] = {}


def walk(
    roots: list[Path],
    extra_text_exts: frozenset[str] = frozenset(),
    max_size: dict[str, int] | None = None,
    stats: WalkStats | None = None,
    known_roots: frozenset[Path] = frozenset(),
    max_entries: int | None = None,
    max_seconds: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[FileEntry]:
    """Yield supported files under the given roots (files or directories).

    Roots should be canonical absolute paths. known_roots are other indexed
    roots; ignore files above a root are read up to the nearest of them.
    With max_entries or max_seconds, raise WalkTruncated once that many
    directory entries were looked at or that much time has passed.
    `should_stop` is asked once per folder; True raises KeyboardInterrupt.
    """
    limits = {**MAX_SIZE, **(max_size or {})}
    stats = stats if stats is not None else WalkStats()
    seen: set[Path] = set()
    deadline = None if max_seconds is None else time.perf_counter() + max_seconds
    count = 0

    def tick(n: int) -> None:
        nonlocal count
        count += n
        if should_stop is not None and should_stop():
            raise KeyboardInterrupt
        if max_entries is not None and count > max_entries:
            raise WalkTruncated(max_entries, by_time=False)
        if deadline is not None and time.perf_counter() > deadline:
            raise WalkTruncated(count, by_time=True)

    def consider(p: Path, st: os.stat_result) -> FileEntry | None:
        stats.seen += 1
        kind = kind_of(p, extra_text_exts)
        if kind is None:
            stats.unsupported += 1
            return None
        try:
            str(p).encode("utf-8")
        except UnicodeEncodeError:  # undecodable bytes in the file name
            stats.ignored += 1
            return None
        if st.st_size > limits[kind] or st.st_size == 0:
            if st.st_size > 0:
                stats.too_large += 1
                if st.st_size <= MAX_SIZE[kind]:
                    # Only over the limit given for this run: keep what is indexed.
                    stats.held.add(str(p))
            return None
        return FileEntry(p, kind, st.st_size, st.st_mtime_ns)

    for root in roots:
        root = Path(os.path.abspath(root))
        try:
            st_root = root.stat()
        except FileNotFoundError:
            stats.missing_roots.append(root)
            continue
        except OSError:
            # A transient stat failure must not make the indexer prune this root.
            stats.unreadable_dirs.append(root)
            continue
        if stat.S_ISREG(st_root.st_mode):
            if root not in seen:
                seen.add(root)
                e = consider(root, st_root)
                if e:
                    yield e
                elif kind_of(root, extra_text_exts) is None:
                    stats.unsupported_roots.append(root)
            continue
        if not stat.S_ISDIR(st_root.st_mode):
            continue
        # Stack of (dir, [(base_dir, spec), ...]) to apply nested ignore files.
        stops = frozenset(r for r in known_roots if r != root)
        above, excluded_by = _ancestor_specs(root, stops)
        if excluded_by is not None:
            stats.ignored_roots.append((root, excluded_by))
        stack: list[tuple[Path, list[tuple[Path, _Rules]]]] = [(root, above)]
        while stack:
            d, specs = stack.pop()
            spec = _load_spec(d)
            if spec is not None:
                specs = [*specs, (d, spec)]
            try:
                tick(1)  # count the directory itself, including an empty one
                with os.scandir(d) as scan:
                    entries = []
                    for e in scan:
                        tick(1)  # stop before consuming and sorting the whole folder
                        entries.append(e)
                entries.sort(key=lambda e: e.name)
                tick(0)  # sorting also counts against the time budget
            except WalkTruncated:
                raise
            except OSError:
                stats.unreadable_dirs.append(d)
                continue
            subdirs: list[Path] = []
            for e in entries:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    stats.unreadable_dirs.append(d / e.name)  # keep what is indexed
                    continue
                if _is_hidden(e.name, st):
                    stats.ignored += 1
                    continue
                if stat.S_ISLNK(st.st_mode):
                    continue  # do not follow symlinks (avoids loops and duplicates)
                is_dir = stat.S_ISDIR(st.st_mode)
                p = d / e.name
                if specs and _is_ignored(p, is_dir, specs):
                    stats.ignored += 1
                    continue
                if is_dir:
                    if e.name in SKIP_DIRS or _is_venv(p):
                        stats.ignored += 1
                        continue
                    subdirs.append(p)
                elif stat.S_ISREG(st.st_mode) and p not in seen:
                    seen.add(p)
                    fe = consider(p, st)
                    if fe:
                        yield fe
            for sd in reversed(subdirs):
                stack.append((sd, specs))
