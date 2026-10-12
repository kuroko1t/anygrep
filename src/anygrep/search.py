"""Brute-force cosine search over the stored vectors."""

from __future__ import annotations

import fnmatch
import os
import re
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import DEFAULT_IMAGE_BUDGET
from .paths import canonical
from .signature import changed_share, signature
from .store import Hit, Store, Vectors
from .thresholds import weak_mask

if TYPE_CHECKING:
    from PIL import Image

    from .embedder import Embedder

_GLOB_CHARS = re.compile(r"[*?\[]")
_RELATIVE = re.compile(r"\.\.?[/\\]")


def clean_text(s: str) -> str:
    """Drop a byte order mark and replace lone surrogates (undecodable bytes in
    argv or stdin), which the tokenizer rejects."""
    return s.replace("\ufeff", "").encode("utf-8", "replace").decode("utf-8")


def query_budget(store: Store) -> int:
    """The budget most indexed images were embedded with (an interrupted or
    partial run can leave a mix), falling back to the stored setting. Queries
    are embedded and judged with it."""
    counts = store.image_budgets()
    if counts:
        return max(counts, key=lambda b: (counts[b], b))
    return int(store.get_meta("image_budget") or DEFAULT_IMAGE_BUDGET)


def image_query(
    store: Store,
    embedder: Embedder,
    image: Image.Image,
    budget: int,
    kinds: Sequence[str] | None,
    prefixes: Sequence[str] | None,
    scope: Callable[[str], bool],
    exclude_path: str | None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """An image query's vector at `budget`, and {budget: vector} for the other
    budgets the images in the scope were embedded with (search()'s
    `image_queries`). ValueError when the image cannot be used (too small)."""
    qvec = embedder.embed_image(image)
    vectors = load_scope(store, kinds, prefixes)
    other_budgets = {
        file_budget
        for path, kind, file_budget in zip(
            vectors.paths, vectors.kinds, vectors.budgets or [], strict=False
        )
        if kind == "image"
        and scope(path)
        and path != exclude_path
        and file_budget is not None
        and file_budget != budget
    }
    image_queries = {}
    try:
        for image_budget in sorted(other_budgets):
            embedder.image_budget = image_budget
            image_queries[image_budget] = embedder.embed_image(image)
    finally:
        embedder.image_budget = budget
    return qvec, image_queries


def path_matcher(pattern: str):
    """Return a predicate for a PATH or a -g/--glob pattern.

    A pattern without wildcards is a directory/file prefix (resolved against
    the current directory). A glob that starts with ~, ./, ../ or is absolute
    is resolved the same way and matched against the full path. Any other glob
    is matched against the full path or any trailing part of it (so "*.md" and
    "notes/*" work).
    """
    if not _GLOB_CHARS.search(pattern):
        return _prefix_matcher(pattern)

    expanded = os.path.expanduser(pattern)
    anchored = expanded != pattern or os.path.isabs(expanded) or _RELATIVE.match(expanded)
    if anchored:
        # Resolve the part before the first wildcard like a folder prefix.
        parts = Path(os.path.abspath(expanded)).parts
        k = next(i for i, x in enumerate(parts) if _GLOB_CHARS.search(x))
        head = canonical(Path(*parts[:k]))
        full = os.path.normcase(head.joinpath(*parts[k:]).as_posix())

        def by_full_glob(p: str) -> bool:
            return fnmatch.fnmatchcase(os.path.normcase(Path(p).as_posix()), full)

        return by_full_glob

    pat = pattern.replace("\\", "/")

    def by_glob(p: str) -> bool:
        q = Path(p).as_posix()
        return fnmatch.fnmatch(q, pat) or fnmatch.fnmatch(q, "*/" + pat.lstrip("/"))

    return by_glob


def scope_matcher(prefixes: Sequence[str], globs: Sequence[str]) -> Callable[[str], bool]:
    """The search scope: paths at or below any of `prefixes` (none: all),
    then matching at least one include glob (if any) and no '!' exclude glob."""
    under = [_prefix_matcher(p) for p in prefixes]
    inc = [path_matcher(g) for g in globs if not g.startswith("!")]
    exc = [path_matcher(g[1:]) for g in globs if g.startswith("!") and len(g) > 1]

    def ok(p: str) -> bool:
        return (
            (not under or any(m(p) for m in under))
            and (not inc or any(m(p) for m in inc))
            and not any(m(p) for m in exc)
        )

    return ok


def _prefix_matcher(path: str) -> Callable[[str], bool]:
    """Paths equal to or below `path` (canonicalised like indexed paths),
    even when its name contains glob characters."""
    base = os.path.normcase(str(canonical(path)))
    prefix = base.rstrip("/\\") + os.sep

    def by_prefix(p: str) -> bool:
        q = os.path.normcase(p)
        return q == base or q.startswith(prefix)

    return by_prefix


# Two files are near-duplicates when they have the same type and either the
# same SHA-256, or both the cosine between their best-matching chunks and the
# cosine between their mean chunk vectors reach the value for their type.
# Calibrated with scripts/measure_duplicates.py (docs/DESIGN.md). Images are
# grouped only when their bytes are the same: two charts or screenshots that
# differ in a few numbers embed almost identically (cosine 0.999), and those
# numbers are often why one of them is searched for.
DUP_MIN_COSINE = {"text": 0.95, "pdf": 0.95}
# Text and PDF files must also have almost the same words: at most this share
# of the word pairs of one may be missing from the other (see signature.py;
# added or removed lines do not count). Monthly payslips or bank statements
# embed almost identically (cosine 0.97) but replace 14 to 36 percent of their
# words (month, amounts), which changes 28 percent or more of the pairs; edited
# copies of a note replace at most 5 percent of their words (under 10 percent
# of the pairs). 0.18 corresponds to about 10 percent of the words. A long
# template where only a few amounts change (a 15-line payslip: 3 to 13 percent
# of the pairs) cannot be told from an edited copy this way and is collapsed;
# the edited copies measured reach 10 percent, so no lower value separates them.
# Piped output and -l do not collapse (cli.py), so no file is hidden there.
DUP_MAX_CHANGED = 0.18
# After the n-th distinct result, look at this many more files for copies of
# the results already shown.
DUP_TAIL = 10
# Stop looking for distinct results after this many files that did not join an
# existing group (n * factor + base), and after this many times as many files
# that did (copies are cheap to skip, but a folder of thousands of identical
# files should not make a search read them all).
DUP_SCAN_FACTOR, DUP_SCAN_BASE = 20, 100
DUP_ABSORB_FACTOR = 10


def is_near_duplicate(
    kind: str,
    sha_a,
    sha_b,
    best_cos: float,
    mean_cos: float,
    sigs: Callable[[], tuple[bytes | None, bytes | None]] | None = None,
) -> bool:
    """`sigs` returns the two files' word signatures; it is only called for
    text and PDF files whose vectors already match."""
    if sha_a is not None and sha_a == sha_b:
        return True
    t = DUP_MIN_COSINE.get(kind)
    if t is None or best_cos < t or mean_cos < t:
        return False
    if sigs is not None and kind in ("text", "pdf"):
        changed = changed_share(*sigs())
        if changed is not None and changed > DUP_MAX_CHANGED:
            return False
    return True


def search(
    store: Store,
    qvec: np.ndarray,
    n: int = 10,
    kinds: Sequence[str] | None = None,
    scope: Callable[[str], bool] | None = None,
    per_file: bool = True,
    exclude_path: str | None = None,
    query_kind: str = "text",
    budget: int = DEFAULT_IMAGE_BUDGET,
    hide_weak: bool = False,
    collapse: bool = False,
    image_queries: dict[int, np.ndarray] | None = None,
    prefixes: Sequence[str] | None = None,
    exact: str | None = None,
) -> list[Hit]:
    """Rank chunks by cosine similarity. Every hit carries a `weak` flag
    (see thresholds.py); with `hide_weak` weak chunks are left out. With
    `collapse` (one result per file only), near-duplicate files are grouped
    under the best-ranked one, in its `similar` list, and further results fill
    up the n distinct ones. All reads happen in one snapshot of the index, so
    a concurrent index run cannot remove rows between them. `image_queries`
    supplies image-query vectors at additional stored image budgets. Without
    a matching query vector, cross-budget image results are marked weak:
    their score distribution has not been calibrated. `prefixes` are the
    folders (or files) of the scope: only their vectors are read from the
    index (`scope` still decides; see load_prefixes()).

    `exact` is the text query: files whose text or name contains it (see
    exact_needle()) come first, best score first, marked `exact` and never
    weak, whatever their score."""
    if n <= 0:
        return []
    with store.snapshot():
        return _search(
            store, qvec, n, kinds, scope, per_file, exclude_path, query_kind, budget,
            hide_weak, collapse, image_queries, prefixes, exact,
        )  # fmt: skip


# Exact matches: a query of at least this many characters (shorter ones
# would match almost everything) is also looked up as a plain string in the
# chunk text and in the file names. Scores are cosines (at most 1), so the
# boost puts exact matches above every other result without changing their
# order among themselves.
EXACT_MIN_CHARS = 3
_EXACT_BOOST = 10.0


def exact_needle(query: str | None) -> str | None:
    """The string exact matching looks for, or None for no exact matching."""
    if query is None:
        return None
    q = query.strip()
    return q if len(q) >= EXACT_MIN_CHARS else None


def _relative_names(paths: Sequence[str], bases: Sequence[str]) -> list[str]:
    """Each path relative to the longest of `bases` it is under, with /
    separators and folded case (its file name when it is a base itself or
    under none)."""
    folded = sorted({os.path.normcase(b.rstrip("/\\")) for b in bases}, key=len, reverse=True)
    out = []
    for p in paths:
        q = os.path.normcase(p)
        rel = None
        for b in folded:
            if q.startswith(b + os.sep):
                rel = p[len(b) + 1 :]
                break
        if rel is None:
            rel = os.path.basename(p)
        out.append(rel.replace(os.sep, "/").casefold())
    return out


def _exact_rows(
    store: Store,
    needle: str,
    v: Vectors,
    file_ok: np.ndarray,
    scores: np.ndarray,
    prefixes: Sequence[str] | None,
    per_file: bool,
) -> np.ndarray:
    """Rows to rank as exact matches: per file the best chunk containing the
    needle, or its best chunk when only its name does (with per_file false,
    every chunk containing it)."""
    rows = np.zeros(len(v.ids), bool)
    ids = store.exact_chunks(needle)
    if ids is None:  # no FTS5 in this SQLite: meaning only
        return rows
    text_rows = (
        np.isin(v.ids, np.fromiter(ids, np.int64, len(ids))) if ids else rows.copy()
    ) & file_ok[v.file_of]
    fold = needle.casefold().replace("\\", "/")
    bases = list(prefixes) if prefixes else store.roots()
    names = _relative_names(v.paths, bases)
    by_name = np.fromiter(
        (ok and fold in name for ok, name in zip(file_ok, names, strict=True)), bool, len(names)
    )
    with_text = np.zeros(len(v.paths), bool)
    with_text[v.file_of[text_rows]] = True
    cand = np.flatnonzero(text_rows | (by_name & ~with_text)[v.file_of])
    if len(cand) == 0:
        return rows
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    _, first = np.unique(v.file_of[cand], return_index=True)
    rows[cand[first]] = True
    return rows if per_file else rows | text_rows


def load_prefixes(prefixes: Sequence[str] | None) -> list[str] | None:
    """The scope folders in their stored (canonical) form, for reading only
    their vectors; None (read everything) without a scope, or where paths are
    compared case-insensitively (Windows): SQLite compares them as stored."""
    if not prefixes or os.path.normcase("A") != "A":
        return None
    return [str(canonical(p)) for p in prefixes]


def load_scope(store: Store, kinds, prefixes: Sequence[str] | None) -> Vectors:
    """The vectors a search with these scope folders looks at."""
    return store.load_matrix(kinds, load_prefixes(prefixes))


class _Ranked:
    """Chunk rows by descending score, ties by row (np.argsort(-scores,
    kind='stable')), computed lazily: a search looks at the first few dozen
    files, so sorting every chunk of a large index is mostly wasted. The
    order is extended (doubling) when more rows are read."""

    START = 256

    def __init__(self, scores: np.ndarray, want: int) -> None:
        self.scores = scores
        self.n = len(scores)
        self.order = np.zeros(0, np.int64)
        self._extend(max(self.START, want))

    def _extend(self, k: int) -> None:
        if 2 * k >= self.n or not np.all(np.isfinite(self.scores) | np.isneginf(self.scores)):
            self.order = np.argsort(-self.scores, kind="stable")
            return
        # Every row scoring at least the k-th best: exactly the first rows of
        # the full stable order (ties at the boundary are all included).
        t = np.partition(self.scores, self.n - k)[self.n - k]
        idx = np.flatnonzero(self.scores >= t)
        self.order = idx[np.argsort(-self.scores[idx], kind="stable")]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, pos: int) -> int:
        if pos >= len(self.order):
            self._extend(max(2 * len(self.order), pos + 1))
        return int(self.order[pos])


def _search(
    store, qvec, n, kinds, scope, per_file, exclude_path, query_kind, budget, hide_weak,
    collapse, image_queries, prefixes=None, exact=None,
) -> list[Hit]:  # fmt: skip
    v = load_scope(store, kinds, prefixes)
    if len(v.ids) == 0:
        return []
    scores = v.mat @ np.asarray(qvec, np.float32)
    row_kinds = v.row_kinds()
    # Weak is decided per result from the query and the shipped reference set,
    # independent of what is indexed and of the scope.
    weak = weak_mask(scores, row_kinds, query_kind, budget, qvec)
    if v.budgets is not None:
        file_budgets = np.asarray([b if b is not None else budget for b in v.budgets])
        row_budgets = file_budgets[v.file_of]
        image_rows = row_kinds == "image"
        for result_budget in np.unique(row_budgets[image_rows]):
            result_budget = int(result_budget)
            if result_budget == budget:
                continue
            sel = image_rows & (row_budgets == result_budget)
            image_q = qvec
            if query_kind == "image":
                image_q = (image_queries or {}).get(result_budget)
                if image_q is None:
                    weak[sel] = True
                    continue
                scores[sel] = v.mat[sel] @ np.asarray(image_q, np.float32)
            weak[sel] = weak_mask(scores[sel], row_kinds[sel], query_kind, result_budget, image_q)
    keep = ~weak if hide_weak else np.ones(len(v.ids), bool)
    # Paths are tested once per file, not once per chunk.
    file_ok = np.ones(len(v.paths), bool)
    if scope is not None or exclude_path:
        file_ok = np.fromiter(
            ((scope is None or scope(p)) and p != exclude_path for p in v.paths),
            bool,
            len(v.paths),
        )
    exact_rows = np.zeros(len(v.ids), bool)
    needle = exact_needle(exact) if query_kind == "text" else None
    if needle is not None:
        exact_rows = _exact_rows(store, needle, v, file_ok, scores, prefixes, per_file)
        keep |= exact_rows
        weak &= ~exact_rows
    keep &= file_ok[v.file_of]
    rank = np.where(keep, scores + _EXACT_BOOST * exact_rows, -np.inf)
    order = _Ranked(rank, 4 * n)
    if per_file and collapse:
        groups = _collapsed(store, order, rank, v, n)
    else:
        groups = []
        seen_files: set[int] = set()
        for pos in range(len(order)):
            i = order[pos]
            if not np.isfinite(rank[i]):
                break
            if per_file:
                f = int(v.file_of[i])
                if f in seen_files:
                    continue
                seen_files.add(f)
            groups.append([int(i)])
            if len(groups) >= n:
                break
    details = store.chunk_details(int(v.ids[i]) for g in groups for i in g)

    def hit(i: int) -> Hit | None:
        d = details.get(int(v.ids[i]))
        if d is None:  # removed since the vectors were read
            return None
        page, ls, le, text = d
        f = int(v.file_of[i])
        return Hit(
            float(scores[i]), v.paths[f], v.kinds[f], page, ls, le, text, int(v.ids[i]),
            weak=bool(weak[i]), exact=bool(exact_rows[i]),
        )  # fmt: skip

    hits = []
    for g in groups:
        h = hit(g[0])
        if h is None:
            continue
        h.similar = [x for x in map(hit, g[1:]) if x is not None]
        hits.append(h)
    return hits


def _collapsed(store, order, scores, v: Vectors, n: int) -> list[list[int]]:
    """The best chunk of each file in score order, grouped into near-duplicates:
    [[representative row, member rows...], ...], at most n groups. Files are
    read from the index in batches, so only the files looked at cost a query."""
    groups: list[list[int]] = []
    info: dict[str, tuple] = {}
    sigs: dict[str, bytes | None] = {}
    seen: set[int] = set()
    limit = n * DUP_SCAN_FACTOR + DUP_SCAN_BASE
    pos, after_full, absorbed = 0, 0, 0
    paths, kinds = v.paths, v.kinds

    def sig_of(path: str) -> bytes | None:
        if path not in sigs:
            s = info.get(path, (None, None, None))[2]
            if s is None:  # indexed before signatures were stored
                text = store.file_texts([path]).get(path)
                s = signature([text]) if text else None
            sigs[path] = s
        return sigs[path]

    while len(seen) - absorbed < limit and absorbed < limit * DUP_ABSORB_FACTOR:
        want = max(2 * n, 20) if len(groups) < n else DUP_TAIL - after_full
        new: list[int] = []
        while len(new) < want and pos < len(order):
            i = int(order[pos])
            pos += 1
            if not np.isfinite(scores[i]):
                pos = len(order)
                break
            f = int(v.file_of[i])
            if f not in seen:
                seen.add(f)
                new.append(i)
        if not new:
            break
        info.update(store.file_info(paths[v.file_of[i]] for i in new))
        for i in new:
            if len(groups) >= n:
                after_full += 1
            fi = int(v.file_of[i])
            sha, mean, _sig = info.get(paths[fi], (None, None, None))
            for g in groups:
                r = g[0]
                fr = int(v.file_of[r])
                rsha, rmean, _rsig = info.get(paths[fr], (None, None, None))
                if (
                    kinds[fr] == kinds[fi]
                    and mean is not None
                    and rmean is not None
                    and is_near_duplicate(
                        kinds[fi],
                        sha,
                        rsha,
                        float(v.mat[i] @ v.mat[r]),
                        float(mean @ rmean),
                        lambda a=paths[fi], b=paths[fr]: (sig_of(a), sig_of(b)),
                    )
                ):
                    g.append(i)
                    absorbed += 1
                    break
            else:
                if len(groups) < n:
                    groups.append([i])
            if len(groups) >= n and after_full >= DUP_TAIL:
                return groups
    return groups


# C0 and C1 control characters, DEL and bidirectional overrides. Printed to a
# terminal as they are, text from an indexed file (or a file name) could set
# the clipboard or the window title, clear the screen or fake result lines.
_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def printable(s: str) -> str:
    """s with control characters replaced by '?', for terminal output."""
    return _CONTROL.sub("?", s)


_CJK_RUN = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]+")
_KANA_ONLY = re.compile(r"[\u3040-\u309f\u30fc]+")


def _cols(ch: str) -> int:
    """Terminal columns a character takes (CJK and other wide characters: 2)."""
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def display_width(s: str) -> int:
    return sum(_cols(c) for c in s)


def _query_terms(query: str) -> list[str]:
    """Words to look for in a chunk, longest first. Japanese and Chinese have
    no spaces, so their runs are also tried as pieces of 2 to 4 characters
    (pieces that are only kana, such as particles, are left out)."""
    q = query.lower()
    terms = set(re.findall(r"\w{3,}", q))
    for run in _CJK_RUN.findall(q):
        for n in (4, 3, 2):
            for i in range(len(run) - n + 1):
                piece = run[i : i + n]
                if not _KANA_ONLY.fullmatch(piece):
                    terms.add(piece)
    return sorted(terms, key=lambda t: (-len(t), q.find(t)))


def snippet(text: str | None, query: str | None, width: int = 160) -> str:
    """A one-line excerpt of at most `width` terminal columns, centred on the
    longest query word found."""
    if not text:
        return ""
    flat = printable(" ".join(text.split()))
    if display_width(flat) <= width:
        return flat
    pos = -1
    if query:
        low = flat.lower()
        for w in _query_terms(query):
            pos = low.find(w)
            if pos >= 0:
                break
    if pos < 0:
        end, cols = 0, 0
        while end < len(flat) and cols + _cols(flat[end]) <= width - 3:
            cols += _cols(flat[end])
            end += 1
        return flat[:end].rstrip() + "..."
    # Leave room for an ellipsis on each side so the line fits in `width`.
    room = width - 6
    start, cols = pos, 0
    while start > 0 and cols + _cols(flat[start - 1]) <= room // 3:
        start -= 1
        cols += _cols(flat[start])
    end = start
    cols = 0
    while end < len(flat) and cols + _cols(flat[end]) <= room:
        cols += _cols(flat[end])
        end += 1
    while start > 0 and cols + _cols(flat[start - 1]) <= room:  # near the end: fill
        start -= 1
        cols += _cols(flat[start])
    s = flat[start:end].strip()
    return ("..." if start > 0 else "") + s + ("..." if end < len(flat) else "")
