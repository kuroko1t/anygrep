"""SQLite index: files, chunks and their embedding vectors.

Vectors are L2-normalised 768-dim embeddings stored as float16 blobs.
Search is brute-force cosine similarity with numpy, which is fast enough
for hundreds of thousands of chunks.
"""

from __future__ import annotations

import contextlib
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .config import EMBED_DIM, MODEL_NAME, MODEL_REVISION
from .errors import IndexMismatchError

# Schema 2 added the FTS table for exact matches. anygrep 0.2 and earlier
# would change the chunks without it, so they must refuse such an index;
# a schema 1 index is moved to 2 by the first writer that opens it.
SCHEMA_VERSION = 2
OLDEST_SCHEMA = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS roots (path TEXT PRIMARY KEY, added_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT,
    budget INTEGER,
    status TEXT NOT NULL,
    note TEXT,
    indexed_at REAL NOT NULL,
    sig BLOB,
    gen INTEGER
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    page INTEGER,
    line_start INTEGER,
    line_end INTEGER,
    text TEXT,
    vec BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(file_id);
"""


# Exact matching (search.py): an FTS5 index with the trigram tokenizer over
# the folded chunk text (fold(): NFKC and case folded, so the tokenizer
# itself compares bytes), kept in the same transactions as the chunks. It
# stores no copy of the text (content='') and no positions (detail='none'):
# a lookup finds the chunks holding every trigram of the folded query, and
# each one is then checked against its text. Every chunk has a row, also
# those without text (images). Removing a row needs the folded text it was
# added with, so the format records the Unicode version of the folding: an
# index folded with another one is filled in again.
FTS_TABLE = "chunks_fts"
FTS_SCHEMA = (
    f"CREATE VIRTUAL TABLE {FTS_TABLE} USING fts5(text, content='', "
    "tokenize='trigram case_sensitive 1', detail='none')"
)
FTS_FORMAT = f"2 {unicodedata.unidata_version}"
# Chunks whose id is in (fts_pos, fts_end] (meta) are not in the FTS index
# yet: an index made before it existed, filled in by the next update.
FTS_BATCH = 2000
# At most this many trigrams of a query go into the FTS lookup; the check
# against the text makes the result exact anyway.
FTS_MAX_TERMS = 32
_fts_ok: bool | None = None


def fold(text: str) -> str:
    """The form exact matching compares: Unicode NFKC (full-width letters
    and digits become plain ones, accents are composed) and case folded."""
    if text.isascii():
        return text.lower()
    return unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).casefold())


def _sql_fold(text: str | None) -> str | None:
    return None if text is None else fold(text)


# What continues a word around a short query (word_pattern()): a letter or
# digit of any script, or a combining mark, except Chinese, Japanese and
# Korean characters, which are written without spaces next to Latin words
# ("runを実行"). The underscore and the hyphen separate words, as in file
# names ("run_script.sh", "run-script").
_CJK = (
    "\u3005\u3006\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    "\uac00-\ud7af\U00020000-\U0003134f"
)
_WORD_CHAR = rf"(?:[^\W_{_CJK}]|[\u0300-\u036f])"


def word_pattern(needle: str) -> re.Pattern[str]:
    """Matches `needle` (folded) as a whole word: not next to a letter,
    digit or combining mark of any script other than Chinese, Japanese or
    Korean. "run" matches "run_script", "run-script" and "runを", not
    "runé", "érun" or "run2"."""
    return re.compile(rf"(?<!{_WORD_CHAR}){re.escape(needle)}(?!{_WORD_CHAR})")


def _fts_query(needle: str) -> str:
    """An FTS5 query for the chunks holding every trigram of `needle` (3 or
    more characters, folded): non-overlapping ones that cover it, each a
    quoted string, so nothing in the query is FTS syntax."""
    starts = list(range(0, len(needle) - 2, 3))
    if starts[-1] != len(needle) - 3:
        starts.append(len(needle) - 3)
    if len(starts) > FTS_MAX_TERMS:
        step = len(starts) / FTS_MAX_TERMS
        starts = [starts[int(i * step)] for i in range(FTS_MAX_TERMS)] + [starts[-1]]
    terms = dict.fromkeys(needle[s : s + 3] for s in starts)
    return " AND ".join('"' + t.replace('"', '""') + '"' for t in terms)


def fts_available(db: sqlite3.Connection | None = None) -> bool:
    """True when this SQLite has FTS5 with the trigram tokenizer (3.34+)."""
    global _fts_ok
    if _fts_ok is None:
        probe = db or sqlite3.connect(MEMORY)
        try:
            probe.execute(
                "CREATE VIRTUAL TABLE temp._anygrep_probe USING fts5(x, tokenize='trigram')"
            )
            probe.execute("DROP TABLE temp._anygrep_probe")
            _fts_ok = True
        except sqlite3.Error:
            _fts_ok = False
        finally:
            if db is None:
                probe.close()
    return _fts_ok


def _corrupt(e: sqlite3.DatabaseError, fts_write: bool = False) -> bool:
    """True for an error that says the data is damaged, not that the
    database is busy, locked or otherwise unusable for now. With
    `fts_write` (a statement that writes only the FTS index from the chunk
    rows), a constraint failure counts too: FTS5 reports some damage to its
    own tables that way, and the rows it is given (chunk ids and text) can
    break no constraint of an intact index. Constraint failures of the
    other tables are never taken for damage. FTS5 reports damage to its
    settings as an error of its own ("invalid fts5 file format")."""
    if type(e) is sqlite3.DatabaseError:
        return True
    if isinstance(e, sqlite3.OperationalError) and str(e).startswith("invalid fts5 file format"):
        return True
    if not fts_write or not isinstance(e, sqlite3.IntegrityError):
        return False
    name = getattr(e, "sqlite_errorname", None)  # Python 3.11+
    return name is None or name.startswith("SQLITE_CONSTRAINT")


def _create_private(path: Path) -> None:
    """Create an empty index file readable only by the user (the index holds
    the text of every note). SQLite gives its -wal and -shm files the same
    permissions. A new data folder is made private as well."""
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700)
    if os.name == "posix":
        with contextlib.suppress(FileExistsError):
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))


def _make_private(path: Path) -> None:
    """Remove group and other access from an existing index and its files.
    The -wal and -shm files are made writable again when the index is: a
    read-only open of an index whose file was read-only (--no-index) leaves
    them read-only, and SQLite then refuses every write."""
    if os.name != "posix":
        return
    writable = os.access(path, os.W_OK)
    for s in ("", "-wal", "-shm"):
        p = path.with_name(path.name + s)
        with contextlib.suppress(OSError):
            mode = p.stat().st_mode
            new = mode & 0o700
            if s and writable:
                new |= 0o600
            if new != mode & 0o777:
                p.chmod(new)


@dataclass
class FileRow:
    id: int
    path: str
    kind: str
    size: int
    mtime_ns: int
    sha256: str | None
    budget: int | None
    status: str
    note: str | None = None
    gen: int | None = None  # processing generation (None: before they were recorded)


@dataclass
class ChunkData:
    ord: int
    vec: np.ndarray | None  # None until the chunk is embedded
    text: str | None = None
    page: int | None = None
    line_start: int | None = None
    line_end: int | None = None


@dataclass
class Hit:
    score: float
    path: str
    kind: str
    page: int | None
    line_start: int | None
    line_end: int | None
    text: str | None
    chunk_id: int
    weak: bool = False
    # The file's text or name contains the query (search.py, exact matches).
    exact: bool = False
    # Weak, but less than search.NEAR_MARGIN under the cutoff.
    near: bool = False
    # Near-duplicate files collapsed under this one (see search.py).
    similar: list[Hit] = field(default_factory=list)


def vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float16).tobytes()


@dataclass
class Vectors:
    """All searchable chunk vectors, with their files listed once."""

    ids: np.ndarray  # [N] chunk ids
    mat: np.ndarray  # [N, D] float32, L2-normalised rows
    file_of: np.ndarray  # [N] index into paths/kinds
    paths: list[str]
    kinds: list[str]
    budgets: list[int | None] | None = None  # per file, aligned with paths/kinds

    def row_kinds(self) -> np.ndarray:
        return np.asarray(self.kinds or [""])[self.file_of]


# Rows read from SQLite per fetch while loading vectors.
_FETCH_ROWS = 4096


def readonly_uri(path: os.PathLike[str] | str, pure=None) -> str:
    """A SQLite URI that opens `path` read-only. `pure` is a PurePath class
    (tests pass PureWindowsPath); characters such as ? and # are escaped."""
    p = (pure or Path)(path)
    if not p.is_absolute():
        p = Path(os.path.abspath(p))
    uri = p.as_uri()
    if not uri.startswith("file:///"):
        # A UNC path (\\server\share\...): SQLite wants an empty authority.
        uri = "file:////" + uri[len("file://") :]
    return uri + "?mode=ro"


MEMORY = ":memory:"


def lock_path(index: Path) -> Path:
    """The side file whose lock marks a running writer (see Store.try_write_lock)."""
    index = Path(index).expanduser().resolve()
    return index.with_name(index.name + "-lock")


def acquire_writer_lock(index: Path) -> sqlite3.Connection | None:
    """Lock the side file of `index` without waiting; None when another
    process holds it. An exclusive SQLite lock on an empty file works the
    same on POSIX and Windows, and the operating system releases it when the
    process ends. Unlike the short transactions of the index itself, it is
    held for a whole run. Close the connection to release it."""
    path = lock_path(index)
    _create_private(path)
    lock = sqlite3.connect(str(path), timeout=0, isolation_level=None)
    try:
        lock.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as e:
        lock.close()
        if "locked" in str(e) or "busy" in str(e):
            return None
        raise
    return lock


class Store:
    def __init__(
        self, path: Path | str, create: bool = True, readonly: bool = False, shared: bool = False
    ) -> None:
        """Open the index at `path` (Store.MEMORY: a throwaway empty one).
        With `readonly` nothing is ever written: no schema, no meta, and
        SQLite refuses writes. With `shared` the connection may be used from
        other threads; the caller makes sure only one uses it at a time."""
        memory = str(path) == MEMORY
        self.path = Path(path) if memory else Path(path).expanduser().resolve()
        self.readonly = readonly
        exists = memory or self.path.exists()
        if (not create or readonly) and not exists:
            raise FileNotFoundError(str(self.path))
        if not exists:
            _create_private(self.path)
        elif create and not readonly and not memory:
            _make_private(self.path)
        if memory:
            self.db = sqlite3.connect(MEMORY, isolation_level=None)
        elif readonly:
            self.db = sqlite3.connect(
                readonly_uri(self.path), uri=True, isolation_level=None, timeout=30
            )
        else:
            self.db = sqlite3.connect(
                str(self.path), isolation_level=None, timeout=30, check_same_thread=not shared
            )
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.create_function("anygrep_fold", 1, _sql_fold, deterministic=True)
        self._exact_cache: tuple[tuple, frozenset[int]] | None = None
        self._vec_cache: tuple[tuple, Vectors] | None = None
        self._lock: sqlite3.Connection | bool | None = None
        tables = {
            r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not tables and readonly:
            self.db.close()
            raise IndexMismatchError(f"{self.path} is not an anygrep index.")
        if not tables:
            # A new index: tables and settings in one transaction, so a crash
            # cannot leave an index that every later run rejects.
            with self.transaction():
                for stmt in SCHEMA.split(";"):
                    if stmt.strip():
                        self.db.execute(stmt)
                self.set_meta("schema_version", str(SCHEMA_VERSION))
                self.set_meta("model", MODEL_NAME)
                self.set_meta("model_revision", MODEL_REVISION)
                self.set_meta("dim", str(EMBED_DIM))
                self.set_meta("created_at", str(time.time()))
                self.set_meta("processing_generation", str(config.PROCESSING_GENERATION))
                if fts_available(self.db):
                    self.db.execute(FTS_SCHEMA)
                    self.set_meta("fts_format", FTS_FORMAT)
        else:
            self._check_existing(tables)
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(files)")}
        # Columns added after anygrep 0.1.0 (sig) and 0.2.0 (gen). A read-only
        # open of an older index reads NULL for them.
        for col, decl in (("sig", "BLOB"), ("gen", "INTEGER")):
            if col not in cols and not readonly:
                with contextlib.suppress(sqlite3.OperationalError):
                    self.db.execute(f"ALTER TABLE files ADD COLUMN {col} {decl}")
        self._sig_col = "f.sig" if "sig" in cols or self._has_column("files", "sig") else "NULL"
        self._gen_col = "gen" if "gen" in cols or self._has_column("files", "gen") else "NULL"
        self.fts = False  # chunk writes keep the FTS index up to date
        self._fts_setup()

    def _fts_setup(self) -> None:
        """Decide whether this connection keeps and reads the FTS index. A
        writer moves an index of schema 1 to this one, and creates the table
        in an index made before it existed or with another format (filled in
        later by fill_fts()); a writer whose SQLite lacks FTS5 leaves it out
        of date, so it records that the index must be rebuilt."""
        has_table = self._has_table(FTS_TABLE)
        supported = fts_available(self.db)
        if self.readonly:
            self.fts = has_table and supported and self._fts_current()
            return
        with contextlib.suppress(sqlite3.OperationalError):  # busy: next time
            if self.get_meta("schema_version") != str(SCHEMA_VERSION):
                with self.transaction():
                    self.set_meta("schema_version", str(SCHEMA_VERSION))
            if supported and (not has_table or self.get_meta("fts_format") != FTS_FORMAT):
                with self.transaction():
                    self._fts_drop()
                    self.db.execute(FTS_SCHEMA)
                    self.set_meta("fts_format", FTS_FORMAT)
                    self.delete_meta("fts_reset")
                    self._fts_mark_gap(0)
                has_table = True
            elif not supported and has_table and self.get_meta("fts_reset") is None:
                with self.transaction():
                    self.set_meta("fts_reset", "1")
            if supported and has_table:
                self._fts_apply_reset()
        self.fts = supported and has_table and self._fts_current()

    def _fts_current(self) -> bool:
        """The FTS table holds the folded text of every chunk outside the gap."""
        return self.get_meta("fts_format") == FTS_FORMAT and self.get_meta("fts_reset") is None

    def _fts_apply_reset(self) -> None:
        """Empty an FTS index that a writer could not keep up to date, to be
        filled in again from the chunks."""
        if self.get_meta("fts_reset") is None:
            return
        with self.transaction():
            if self.get_meta("fts_reset") is None:
                return
            # A new table rather than 'delete-all': the old one may be damaged.
            self._fts_drop()
            self.db.execute(FTS_SCHEMA)
            self._fts_mark_gap(0)
            self.delete_meta("fts_reset")
        self.fts = True

    def _fts_drop(self) -> None:
        """Drop the FTS table, also when it is too damaged to be opened (its
        structure record or settings): those are then put back as an empty
        table has them, which lets SQLite open and drop it. Call inside a
        transaction."""
        if not self._has_table(FTS_TABLE):
            return
        self.db.execute("SAVEPOINT anygrep_fts_drop")
        try:
            self.db.execute(f"DROP TABLE {FTS_TABLE}")
        except sqlite3.DatabaseError as e:
            if not self.db.in_transaction:
                raise
            self.db.execute("ROLLBACK TO anygrep_fts_drop")
            self.db.execute("RELEASE anygrep_fts_drop")
            if not _corrupt(e):
                raise
        else:
            self.db.execute("RELEASE anygrep_fts_drop")
            return
        fresh = "anygrep_fts_fresh"
        self.db.execute(f"DROP TABLE IF EXISTS temp.{fresh}")
        self.db.execute(FTS_SCHEMA.replace(FTS_TABLE, f"temp.{fresh}", 1))
        try:
            for part in ("data", "idx", "docsize", "config"):
                if self._has_table(f"{FTS_TABLE}_{part}"):
                    self.db.execute(f"DELETE FROM main.{FTS_TABLE}_{part}")
            for part in ("data", "config"):
                if self._has_table(f"{FTS_TABLE}_{part}"):
                    self.db.execute(
                        f"INSERT INTO main.{FTS_TABLE}_{part} SELECT * FROM temp.{fresh}_{part}"
                    )
        finally:
            self.db.execute(f"DROP TABLE temp.{fresh}")
        self.db.execute(f"DROP TABLE {FTS_TABLE}")

    def _fts_check(self) -> None:
        """Look up the first and the last chunk the FTS index holds by a
        trigram of their own text. A lookup that fails, or that misses the
        chunk (a damaged structure record can make every lookup find
        nothing, without an error), records that the index must be made
        again. Call in a writer, outside a transaction."""
        if not self._fts_current():
            return
        live, more = self._fts_live()
        for order in ("ASC", "DESC"):
            row = self.db.execute(
                f"SELECT id, text FROM chunks WHERE text IS NOT NULL AND {live} "
                f"ORDER BY id {order} LIMIT 1",
                more,
            ).fetchone()
            if row is None:
                return
            folded = fold(row[1])
            if len(folded) < 3:
                continue
            try:
                ok = self.db.execute(
                    f"SELECT 1 FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ? AND rowid = ?",
                    (_fts_query(folded[:3]), row[0]),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                if not _corrupt(e):
                    raise
                ok = None
            if ok is None:
                self._fts_damaged()
                return

    def _fts_damaged(self) -> None:
        """Record that a read found the FTS index damaged, so that the next
        update makes it again (lookups read the chunk text until then). A
        read-only store, or one another process is writing, leaves that to
        the check of the next fill_fts()."""
        if self.readonly or self.get_meta("fts_reset") is not None:
            return
        if self.db.in_transaction:
            self.set_meta("fts_reset", "1")
            self.fts = False
            return
        (wait,) = self.db.execute("PRAGMA busy_timeout").fetchone()
        try:
            self.db.execute("PRAGMA busy_timeout=0")
            with self.transaction():
                self.set_meta("fts_reset", "1")
            self.fts = False
        except sqlite3.OperationalError:  # busy: the next check finds it
            pass
        finally:
            self.db.execute(f"PRAGMA busy_timeout={int(wait)}")

    def _has_table(self, name: str) -> bool:
        return (
            self.db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()
            is not None
        )

    def _fts_mark_gap(self, pos: int) -> None:
        """Record that the chunks after id `pos`, up to the last one now, are
        not in the FTS index. Call inside a transaction."""
        (end,) = self.db.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()
        if end > pos:
            self.set_meta("fts_pos", str(pos))
            self.set_meta("fts_end", str(end))
        else:
            self.delete_meta("fts_pos")
            self.delete_meta("fts_end")

    def fts_gap(self) -> tuple[int, int] | None:
        """(pos, end): chunks with pos < id <= end are not in the FTS index
        yet; None when every chunk is."""
        pos, end = self.get_meta("fts_pos"), self.get_meta("fts_end")
        if pos is None or end is None or int(pos) >= int(end):
            return None
        return int(pos), int(end)

    def exact_state(self) -> str:
        """ "ready", "building" (an update fills in the older chunks) or
        "unavailable" (this SQLite has no FTS5 trigram tokenizer)."""
        if not fts_available(self.db):
            return "unavailable"
        if not self.fts or not self._fts_current() or self.fts_gap():
            return "building"
        return "ready"

    def _fts_live(self) -> tuple[str, list[int]]:
        """An SQL condition on chunk `id` true for the chunks that are (or
        are to be) in the FTS index now: those outside the gap."""
        gap = self.fts_gap()
        if gap is None:
            return "1", []
        return "(id <= ? OR id > ?)", list(gap)

    def fill_fts(self, max_seconds: float | None = None, quiet: bool = True) -> bool:
        """Add the chunks of the gap to the FTS index, in short transactions
        (an interrupted run keeps what is done). True when none is left.
        When a batch finds the index damaged, its position is kept, and the
        index is made again from the chunk text (chunks and vectors stay);
        a second failure in the same run leaves that for the next one."""
        if self.readonly or not fts_available(self.db) or not self._has_table(FTS_TABLE):
            return True
        from tqdm import tqdm

        start = time.monotonic()
        self._fts_check()
        for _attempt in range(2):
            self._fts_apply_reset()
            if not self._fts_writes():
                return self.get_meta("fts_reset") is None
            gap = self.fts_gap()
            if gap is None:
                return True
            pos, end = gap
            bar = tqdm(
                total=end - pos, disable=quiet, desc="Exact-match index", file=sys.stderr,
                delay=1.0, dynamic_ncols=True, leave=False,
            )  # fmt: skip
            try:
                while pos < end:
                    if max_seconds is not None and time.monotonic() - start > max_seconds:
                        return False
                    with self.transaction():
                        if not self._fts_writes():
                            break
                        gap = self.fts_gap()
                        if gap is None:
                            return True
                        pos, end = gap
                        r = self.db.execute(
                            "SELECT id FROM chunks WHERE id > ? AND id <= ? ORDER BY id "
                            "LIMIT 1 OFFSET ?",
                            (pos, end, FTS_BATCH - 1),
                        ).fetchone()
                        upto = end if r is None else r[0]
                        self._fts_exec(
                            f"INSERT INTO {FTS_TABLE}(rowid, text) "
                            "SELECT id, anygrep_fold(text) FROM chunks WHERE id > ? AND id <= ?",
                            (pos, upto),
                        )
                        if not self.fts:  # damaged: rebuilt below, from the start
                            break
                        if upto >= end:
                            self.delete_meta("fts_pos")
                            self.delete_meta("fts_end")
                        else:
                            self.set_meta("fts_pos", str(upto))
                    bar.update(upto - pos)
                    pos = upto
                else:
                    return self.fts_gap() is None
            finally:
                bar.close()
        return False

    def _fts_writes(self) -> bool:
        """Whether chunk writes must update the FTS index. Call inside a
        transaction: another process may have created the table since this
        one opened the index."""
        if not self.fts and not self.readonly and fts_available(self.db):
            self.fts = self._has_table(FTS_TABLE) and self._fts_current()
        elif self.fts and self.get_meta("fts_reset") is not None:
            self.fts = False
        return self.fts

    def _fts_exec(self, sql: str, params: Iterable) -> None:
        """Run a write to the FTS index. When SQLite finds the index damaged
        (rows changed by a program that did not keep it), the chunks are
        still written, and the index is rebuilt from them by the next
        update; until then lookups read the chunk text."""
        self.db.execute("SAVEPOINT anygrep_fts")
        try:
            self.db.execute(sql, list(params))
            # FTS5 keeps new rows in memory until a savepoint or the commit
            # and only then finds damage; write them now, so that a failure
            # undoes just this statement instead of failing the commit.
            self.db.execute("SAVEPOINT anygrep_fts_flush")
            self.db.execute("RELEASE anygrep_fts_flush")
        except sqlite3.DatabaseError as e:
            if not self.db.in_transaction:  # SQLite rolled back everything
                raise
            self.db.execute("ROLLBACK TO anygrep_fts")
            self.db.execute("RELEASE anygrep_fts")
            if not _corrupt(e, fts_write=True):
                raise
            self.set_meta("fts_reset", "1")
            self.fts = False
        else:
            self.db.execute("RELEASE anygrep_fts")

    def _fts_delete(self, where: str, params: Iterable) -> None:
        """Take the chunks matching `where` out of the FTS index (before they
        are deleted). Call inside a transaction."""
        if not self._fts_writes():
            return
        live, more = self._fts_live()
        self._fts_exec(
            f"INSERT INTO {FTS_TABLE}({FTS_TABLE}, rowid, text) "
            f"SELECT 'delete', id, anygrep_fold(text) FROM chunks WHERE ({where}) AND {live}",
            [*params, *more],
        )

    def exact_chunks(self, needle: str, word: bool = False) -> set[int] | None:
        """Ids of the chunks whose folded text (see fold()) contains `needle`
        (folded, 3 or more characters); with `word`, as a whole word (see
        word_pattern()). None when this SQLite cannot use the FTS index.
        Chunks the FTS index does not hold (yet) are read directly."""
        if not fts_available(self.db):
            return None
        key = (needle, word, self.db.total_changes, self._data_version())
        if self._exact_cache is not None and self._exact_cache[0] == key:
            return set(self._exact_cache[1])
        if len(needle) < 3:
            return set()
        found = self._exact_lookup(needle, word)
        self._exact_cache = (key, frozenset(found))
        return found

    def _data_version(self) -> int:
        return int(self.db.execute("PRAGMA data_version").fetchone()[0])

    def _exact_lookup(self, needle: str, word: bool) -> set[int]:
        pat = word_pattern(needle) if word else None

        def hit(text: str | None) -> bool:
            if not text:
                return False
            t = fold(text)
            return pat.search(t) is not None if pat else needle in t

        found: set[int] = set()
        scan: tuple[str, list[int]] | None = ("1", [])
        if self.fts and self._fts_current():
            try:
                cand = [
                    r[0]
                    for r in self.db.execute(
                        f"SELECT rowid FROM {FTS_TABLE} WHERE {FTS_TABLE} MATCH ?",
                        (_fts_query(needle),),
                    )
                ]
            except sqlite3.DatabaseError as e:
                if not _corrupt(e):
                    raise
                self._fts_damaged()
            else:
                # One trigram that is the whole query: the lookup is exact.
                if len(needle) == 3 and pat is None:
                    found.update(cand)
                else:
                    for i in range(0, len(cand), 500):
                        part = cand[i : i + 500]
                        q = ",".join("?" * len(part))
                        found.update(
                            cid
                            for cid, text in self.db.execute(
                                f"SELECT id, text FROM chunks WHERE id IN ({q})", part
                            )
                            if hit(text)
                        )
                gap = self.fts_gap()
                scan = ("id > ? AND id <= ?", list(gap)) if gap else None
        if scan is not None:
            cond, params = scan
            found.update(
                cid
                for cid, text in self.db.execute(
                    f"SELECT id, text FROM chunks WHERE text IS NOT NULL AND {cond}", params
                )
                if hit(text)
            )
        return found

    def _check_existing(self, tables: set[str]) -> None:
        """Refuse files that are not anygrep indexes (without writing to
        them) and indexes of another schema or model."""
        version = self.get_meta("schema_version") if "meta" in tables else None
        if version is None or not {"files", "chunks", "roots"} <= tables:
            self.db.close()
            raise IndexMismatchError(f"{self.path} is not an anygrep index.")
        try:
            v = int(version)
        except ValueError:
            v = -1
        if v > SCHEMA_VERSION:
            self.db.close()
            raise IndexMismatchError(
                f"Index {self.path} was created by a newer anygrep (schema {version}; this "
                f"version reads schema {SCHEMA_VERSION}). Upgrade anygrep to use it."
            )
        if not OLDEST_SCHEMA <= v <= SCHEMA_VERSION:
            self.db.close()
            raise IndexMismatchError(
                f"Index {self.path} uses schema {version}, "
                f"this version expects {SCHEMA_VERSION}. Run `anygrep reset` and re-index."
            )
        if self.get_meta("model_revision") != MODEL_REVISION:
            self.db.close()
            raise IndexMismatchError(
                f"Index {self.path} was built with a different model revision. "
                "Run `anygrep reset` and re-index."
            )

    def close(self) -> None:
        self.release_write_lock()
        self.db.close()

    def lock_path(self) -> Path:
        return lock_path(self.path)

    def try_write_lock(self, probe_db: bool = True) -> bool:
        """Take the writer lock without waiting: True when no other anygrep
        process is changing the index. The lock is held until
        release_write_lock() or close(), and the operating system drops it
        when the process ends. With `probe_db`, a writer holding the database
        itself (BEGIN IMMEDIATE, e.g. another tool) also counts as busy."""
        if self.readonly:
            return False
        if self._lock is not None:
            return True
        if str(self.path) != MEMORY:
            lock = acquire_writer_lock(self.path)
            if lock is None:
                return False
            self._lock = lock
        if probe_db and not self._probe_db():
            self.release_write_lock()
            return False
        if self._lock is None:
            self._lock = True
        return True

    def _probe_db(self) -> bool:
        self.db.execute("PRAGMA busy_timeout=0")
        try:
            self.db.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as e:
            if "locked" in str(e) or "busy" in str(e):
                return False
            raise
        finally:
            self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute("ROLLBACK")
        return True

    def release_write_lock(self) -> None:
        lock, self._lock = self._lock, None
        if isinstance(lock, sqlite3.Connection):
            with contextlib.suppress(sqlite3.Error):
                lock.close()

    def _has_column(self, table: str, column: str) -> bool:
        return any(r[1] == column for r in self.db.execute(f"PRAGMA table_info({table})"))

    # -- meta
    def get_meta(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def delete_meta(self, key: str) -> None:
        self.db.execute("DELETE FROM meta WHERE key=?", (key,))

    # -- roots
    def add_roots(self, roots: Iterable[str]) -> None:
        now = time.time()
        self.db.executemany(
            "INSERT OR IGNORE INTO roots(path, added_at) VALUES(?, ?)",
            [(r, now) for r in roots],
        )

    def roots(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT path FROM roots ORDER BY path")]

    def remove_roots(self, roots: Iterable[str]) -> None:
        self.db.executemany("DELETE FROM roots WHERE path=?", [(r,) for r in roots])

    # -- files
    def files_under(self, root: str) -> Iterator[FileRow]:
        """Rows whose path equals root or lies below it."""
        prefix = root.rstrip("/\\") + os.sep
        cur = self.db.execute(
            f"SELECT id, path, kind, size, mtime_ns, sha256, budget, status, note, "
            f"{self._gen_col} FROM files WHERE path = ? OR substr(path, 1, ?) = ?",
            (root, len(prefix), prefix),
        )
        for r in cur:
            yield FileRow(*r)

    def get_file(self, path: str) -> FileRow | None:
        r = self.db.execute(
            f"SELECT id, path, kind, size, mtime_ns, sha256, budget, status, note, "
            f"{self._gen_col} FROM files WHERE path=?",
            (path,),
        ).fetchone()
        return FileRow(*r) if r else None

    # -- processing generations
    def _stale_sql(self, col: str = "") -> tuple[str, list[int]]:
        """An SQL condition true for rows processed by an older generation
        than their kind needs (see config.PROCESSING_GENERATION), and its
        parameters. `col` prefixes the column names ("f.")."""
        gen = f"{col}{self._gen_col}" if self._gen_col != "NULL" else "NULL"
        kinds = ("text", "pdf", "image")
        case = " ".join(f"WHEN '{k}' THEN ?" for k in kinds)
        return (
            f"COALESCE({gen}, 1) < (CASE {col}kind {case} ELSE 0 END)",
            [config.required_generation(k) for k in kinds],
        )

    def is_stale(self, row: FileRow) -> bool:
        """True when the file was processed by an older generation than its
        kind needs: the next update processes it again."""
        return (row.gen or 1) < config.required_generation(row.kind)

    def count_stale(self, prefixes: Iterable[str] | None = None) -> int:
        """Files (at or below any of `prefixes`; None: all) processed by an
        older generation than their kind needs."""
        cond, params = self._stale_sql()
        where, more = _under_sql(prefixes)
        (n,) = self.db.execute(
            f"SELECT COUNT(*) FROM files WHERE {cond}{where}", params + more
        ).fetchone()
        return int(n)

    def image_shas(self, budget: int) -> dict[str, int]:
        """{sha256: file id} of the indexed images embedded at this budget
        by the current processing generation."""
        cond, params = self._stale_sql()
        return dict(
            self.db.execute(
                "SELECT sha256, id FROM files WHERE kind='image' AND status='ok' AND budget=? "
                f"AND sha256 IS NOT NULL AND NOT ({cond})",
                [budget, *params],
            ).fetchall()
        )

    def image_vector(self, file_id: int, sha256: str, budget: int) -> np.ndarray | None:
        """The vector of an indexed image with this content and budget (None
        when the row is gone or changed since image_shas())."""
        cond, params = self._stale_sql("f.")
        r = self.db.execute(
            "SELECT c.vec FROM chunks c JOIN files f ON f.id=c.file_id WHERE f.id=? "
            "AND f.kind='image' AND f.status='ok' AND f.sha256=? AND f.budget=? "
            f"AND NOT ({cond}) LIMIT 1",
            [file_id, sha256, budget, *params],
        ).fetchone()
        return None if r is None else np.frombuffer(r[0], np.float16).astype(np.float32)

    def touch_files(self, rows: Iterable[tuple[int, int, int]]) -> None:
        """Set (file_id, size, mtime_ns) for files whose content did not change."""
        rows = list(rows)
        if rows:
            with self.transaction():
                self.db.executemany(
                    "UPDATE files SET size=?, mtime_ns=? WHERE id=?",
                    [(size, mtime, fid) for fid, size, mtime in rows],
                )

    def move_file(self, file_id: int, path: str, size: int, mtime_ns: int) -> bool:
        """Give a row a new path; False when the row no longer exists."""
        cur = self.db.execute(
            "UPDATE files SET path=?, size=?, mtime_ns=? WHERE id=?",
            (path, size, mtime_ns, file_id),
        )
        return cur.rowcount == 1

    def delete_files(self, ids: Iterable[int]) -> int:
        ids = list(ids)
        with self.transaction():
            for i in range(0, len(ids), 500):
                part = ids[i : i + 500]
                q = ",".join("?" * len(part))
                self._fts_delete(f"file_id IN ({q})", part)
                self.db.execute(f"DELETE FROM chunks WHERE file_id IN ({q})", part)
                self.db.execute(f"DELETE FROM files WHERE id IN ({q})", part)
        return len(ids)

    def transaction(self) -> _Tx:
        return _Tx(self.db)

    def put_file(
        self,
        path: str,
        kind: str,
        size: int,
        mtime_ns: int,
        sha256: str | None,
        budget: int | None,
        status: str,
        note: str | None,
        chunks: list[ChunkData],
        sig: bytes | None = None,
    ) -> int:
        """Replace a file and all its chunks and return its id. Call inside a
        transaction."""
        old = self.db.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
        if old:
            self._fts_delete("file_id=?", (old[0],))
            self.db.execute("DELETE FROM chunks WHERE file_id=?", (old[0],))
            self.db.execute("DELETE FROM files WHERE id=?", (old[0],))
        cur = self.db.execute(
            "INSERT INTO files(path, kind, size, mtime_ns, sha256, budget, status, note, "
            "indexed_at, sig, gen) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                path, kind, size, mtime_ns, sha256, budget, status, note, time.time(), sig,
                config.PROCESSING_GENERATION,
            ),
        )  # fmt: skip
        fid = cur.lastrowid
        assert fid is not None
        self.add_chunks(fid, chunks)
        return fid

    def add_chunks(self, file_id: int, chunks: list[ChunkData]) -> None:
        """Insert chunks of a file. Call inside a transaction."""
        rows = []
        for c in chunks:
            if c.vec is None:
                raise ValueError(f"chunk {c.ord} of file {file_id} has no vector")
            rows.append(
                (file_id, c.ord, c.page, c.line_start, c.line_end, c.text, vec_to_blob(c.vec))
            )
        if not rows:
            return
        fts = self._fts_writes()
        if fts:
            # New rows get ids above the largest one (SQLite's rowid rule).
            (before,) = self.db.execute("SELECT COALESCE(MAX(id), 0) FROM chunks").fetchone()
        self.db.executemany(
            "INSERT INTO chunks(file_id, ord, page, line_start, line_end, text, vec) "
            "VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        if fts:
            live, more = self._fts_live()
            self._fts_exec(
                f"INSERT INTO {FTS_TABLE}(rowid, text) SELECT id, anygrep_fold(text) "
                f"FROM chunks WHERE id > ? AND {live}",
                [before, *more],
            )

    def set_status(self, file_id: int, status: str, note: str | None) -> None:
        self.db.execute("UPDATE files SET status=?, note=? WHERE id=?", (status, note, file_id))

    def resume_point(
        self, path: str, size: int, mtime_ns: int, sha256: str
    ) -> tuple[int, int, str | None] | None:
        """For a file left half-indexed (status 'partial') with this content:
        (file id, number of chunks stored, text of the last one)."""
        cond, params = self._stale_sql()
        r = self.db.execute(
            "SELECT id FROM files WHERE path=? AND status='partial' AND size=? AND mtime_ns=? "
            f"AND sha256=? AND NOT ({cond})",
            [path, size, mtime_ns, sha256, *params],
        ).fetchone()
        if r is None:
            return None
        n, last = self.db.execute(
            "SELECT COUNT(*), (SELECT text FROM chunks WHERE file_id=? ORDER BY ord DESC LIMIT 1) "
            "FROM chunks WHERE file_id=?",
            (r[0], r[0]),
        ).fetchone()
        return r[0], n, last

    # -- stats
    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for kind, status, n in self.db.execute(
            "SELECT kind, status, COUNT(*) FROM files GROUP BY kind, status"
        ):
            out.setdefault(kind, {})[status] = n
        for kind, n in self.db.execute(
            "SELECT f.kind, COUNT(*) FROM chunks c JOIN files f ON f.id=c.file_id GROUP BY f.kind"
        ):
            out.setdefault(kind, {})["chunks"] = n
        return out

    def image_budgets(self) -> dict[int, int]:
        """Number of indexed images per token budget."""
        return dict(
            self.db.execute(
                "SELECT budget, COUNT(*) FROM files WHERE kind='image' AND status='ok' "
                "AND budget IS NOT NULL GROUP BY budget"
            ).fetchall()
        )

    def count_files_under(self, prefixes: Iterable[str]) -> int:
        """Indexed (searchable) files at or below any of the paths."""
        prefixes = list(prefixes)
        if not prefixes:
            return 0
        where, params = _under_sql(prefixes)
        (n,) = self.db.execute(
            f"SELECT COUNT(*) FROM files WHERE status='ok'{where}", params
        ).fetchone()
        return int(n)

    def has_files_under(self, prefixes: Iterable[str]) -> bool:
        return self.count_files_under(prefixes) > 0

    def has_chunks(self) -> bool:
        return self.db.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None

    def problem_files(self, limit: int = 20) -> list[tuple[str, str, str]]:
        return list(
            self.db.execute(
                "SELECT path, status, note FROM files WHERE status != 'ok' "
                "ORDER BY indexed_at DESC LIMIT ?",
                (limit,),
            )
        )

    # -- search
    @contextlib.contextmanager
    def snapshot(self) -> Iterator[None]:
        """Run the reads inside in one read transaction, so a concurrent
        index run cannot replace rows between them (WAL keeps the snapshot)."""
        if self.db.in_transaction:
            yield
            return
        self.db.execute("BEGIN")
        try:
            yield
        finally:
            if self.db.in_transaction:
                self.db.execute("COMMIT")

    def load_matrix(
        self, kinds: Iterable[str] | None = None, prefixes: Iterable[str] | None = None
    ) -> Vectors:
        """The vectors of all indexed files (of the given types; with
        `prefixes`, only of the files at or below one of these paths, compared
        as stored, so case-sensitively). The result is kept until the index
        changes, so repeated searches (the weak-match fallback, interactive
        mode) do not read it again. Rows keep the same relative order with or
        without `prefixes` (files by id, then chunks by id)."""
        kinds = sorted(set(kinds or []))
        prefixes = sorted(set(prefixes or []))
        with self.snapshot():
            where = "WHERE f.status='ok'"
            params: list = list(kinds)
            if kinds:
                where += f" AND f.kind IN ({','.join('?' * len(kinds))})"
            if prefixes:
                clauses = []
                for root in prefixes:
                    prefix = root.rstrip("/\\") + os.sep
                    clauses.append("f.path = ? OR substr(f.path, 1, ?) = ?")
                    params += [root, len(prefix), prefix]
                where += " AND (" + " OR ".join(f"({c})" for c in clauses) + ")"
            self.db.execute("SELECT 1 FROM meta LIMIT 1").fetchall()  # starts the snapshot
            version = self.db.execute("PRAGMA data_version").fetchone()[0]
            key = (tuple(kinds), tuple(prefixes), version, self.db.total_changes)
            if self._vec_cache is not None and self._vec_cache[0] == key:
                return self._vec_cache[1]
            self._vec_cache = None  # let the old matrix go before loading a new one
            files = self.db.execute(
                f"SELECT f.id, f.path, f.kind, f.budget FROM files f {where}", params
            )
            files = files.fetchall()
            index = {f[0]: i for i, f in enumerate(files)}
            (n,) = self.db.execute(
                f"SELECT COUNT(*) FROM chunks c JOIN files f ON f.id=c.file_id {where}", params
            ).fetchone()
            ids = np.zeros(n, np.int64)
            file_of = np.zeros(n, np.int32)
            # float32: numpy converts float16 slowly, so convert once here.
            mat = np.zeros((n, EMBED_DIM), np.float32)
            cur = self.db.execute(
                f"SELECT c.id, c.file_id, c.vec FROM chunks c JOIN files f ON f.id=c.file_id "
                f"{where} ORDER BY f.id, c.id",  # the order of the index scan: no sort
                params,
            )
            i = 0
            while i < n and (rows := cur.fetchmany(_FETCH_ROWS)):
                rows = rows[: n - i]
                j = i + len(rows)
                ids[i:j] = [r[0] for r in rows]
                file_of[i:j] = [index[r[1]] for r in rows]
                mat[i:j] = np.frombuffer(b"".join(r[2] for r in rows), np.float16).reshape(
                    len(rows), -1
                )
                i = j
        out = Vectors(
            ids[:i],
            mat[:i],
            file_of[:i],
            [f[1] for f in files],
            [f[2] for f in files],
            [f[3] for f in files],
        )
        self._vec_cache = (key, out)
        return out

    def load_vectors(
        self, kinds: Iterable[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
        """Return (chunk_ids, matrix [N, D] float32, paths, kinds), one entry
        per chunk (used by the calibration scripts)."""
        v = self.load_matrix(kinds)
        paths = [v.paths[f] for f in v.file_of]
        return v.ids, v.mat, paths, [v.kinds[f] for f in v.file_of]

    def file_info(
        self, paths: Iterable[str]
    ) -> dict[str, tuple[str | None, np.ndarray, bytes | None]]:
        """{path: (sha256, mean of the chunk vectors L2-normalised, word signature)}."""
        paths = list(dict.fromkeys(paths))
        out: dict[str, tuple[str | None, np.ndarray, bytes | None]] = {}
        for i in range(0, len(paths), 500):
            part = paths[i : i + 500]
            q = ",".join("?" * len(part))
            acc: dict[str, tuple[str | None, bytes | None, list[bytes]]] = {}
            for path, sha, sig, blob in self.db.execute(
                f"SELECT f.path, f.sha256, {self._sig_col}, c.vec FROM files f "
                f"JOIN chunks c ON c.file_id=f.id WHERE f.path IN ({q})",
                part,
            ):
                acc.setdefault(path, (sha, sig, []))[2].append(blob)
            for path, (sha, sig, blobs) in acc.items():
                m = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(len(blobs), -1)
                v = m.astype(np.float32).mean(0)
                norm = float(np.linalg.norm(v))
                out[path] = (sha, v / norm if norm else v, sig)
        return out

    def file_texts(self, paths: Iterable[str]) -> dict[str, str | None]:
        """{path: the file's chunk texts joined in order} (None without text)."""
        paths = list(dict.fromkeys(paths))
        acc: dict[str, list[str]] = {}
        for i in range(0, len(paths), 500):
            part = paths[i : i + 500]
            q = ",".join("?" * len(part))
            for path, text in self.db.execute(
                f"SELECT f.path, c.text FROM files f JOIN chunks c ON c.file_id=f.id "
                f"WHERE f.path IN ({q}) ORDER BY f.path, c.ord",
                part,
            ):
                parts = acc.setdefault(path, [])
                if text:
                    parts.append(text)
        return {p: "\n".join(t) if t else None for p, t in acc.items()}

    def chunk_details(self, ids: Iterable[int]) -> dict[int, tuple]:
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        return {
            r[0]: r[1:]
            for r in self.db.execute(
                f"SELECT id, page, line_start, line_end, text FROM chunks WHERE id IN ({q})",
                ids,
            )
        }


def _under_sql(prefixes: Iterable[str] | None, col: str = "") -> tuple[str, list]:
    """(" AND (...)", parameters): rows whose path is at or below one of
    `prefixes`; ("", []) for None (no limit)."""
    if prefixes is None:
        return "", []
    clauses, params = [], []
    for root in prefixes:
        prefix = root.rstrip("/\\") + os.sep
        clauses.append(f"{col}path = ? OR substr({col}path, 1, ?) = ?")
        params += [root, len(prefix), prefix]
    if not clauses:
        return " AND 0", []
    return " AND (" + " OR ".join(f"({c})" for c in clauses) + ")", params


class _Tx:
    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def __enter__(self) -> sqlite3.Connection:
        self.db.execute("BEGIN IMMEDIATE")
        return self.db

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.db.execute("COMMIT")
        elif self.db.in_transaction:
            # SQLite has already rolled back after some errors (a full disk);
            # a second ROLLBACK would hide the original error.
            self.db.execute("ROLLBACK")
