"""Background indexing: one detached, low-priority worker per index.

The worker is `anygrep index --run-job ...` in its own session (POSIX) or
detached process group (Windows). It holds the index's writer lock for its
whole run and commits every few seconds, so searches meanwhile see the files
indexed so far. Its state lives in a JSON file next to the index:

    I-job.json   state (exists while a job starts, runs, or after it died)
    I-jobmutex   short-lived lock for reading and changing the state
    I-job.stop   stop request: the worker stops at the next file
    I-job.log    the worker's output
    I-job.final  the exact counts of a worker that stopped on request

A job counts as alive only while its process runs and the writer lock is
held, so a reused PID or a foreground `anygrep index` holding the lock is
not mistaken for it. The state records the worker's process start time too:
a process with the same PID but another start time is not the worker, and
`index --stop` kills a worker only when its start time matches. A dead job is reported once and cleaned up; it is never
restarted on its own (`anygrep index --background PATH` resumes it).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import traceback
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import is_under
from .store import _create_private, acquire_writer_lock

STATE_VERSION = 1
# A job whose worker has not taken the writer lock yet (it may still be
# importing numpy and onnxruntime) counts as alive for this long.
STARTING_GRACE = 60.0
# The worker waits this long for the writer lock (the search that started it
# may have just released it).
LOCK_WAIT = 60.0
STATE_EVERY = 3.0
# A batch that just ended is saved at once unless the last save is younger
# than this (a run of copied or skipped files ends many "batches" a second).
STATE_MIN = 0.25
STOP_WAIT = 30.0
# Seconds of measured embedding before the remaining time is shown (before
# that it reads "?", like a fresh tqdm bar; no estimate is shown up front).
RATE_MIN_SECONDS = 10.0
ETA_MIN_STEPS = 2
KILL_WAIT = 5.0
LOG_ROTATE_BYTES = 1 << 20
NICE = 10
# Replaced in tests so the worker can run with a fake model.
WORKER_PREFIX = [sys.executable, "-m", "anygrep"]


@dataclass
class JobState:
    version: int = STATE_VERSION
    pid: int = 0
    # The process start time of `pid` (process_start), so a reused PID is
    # not taken for the worker; "" when unknown.
    pid_start: str = ""
    # Set when the job is requested; the worker gets it as --job-id.
    job_id: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    threads: int = 1
    scope: list[str] = field(default_factory=list)
    queue: list[list[str]] = field(default_factory=list)
    phase: str = "starting"  # starting, planning, text, images, failed
    files_total: int = 0
    files_done: int = 0
    text_total: int = 0
    cost_total: float = 0.0
    cost_done: float = 0.0
    embed_seconds: float = 0.0
    # Batches finished: the first one's speed says little (one large file,
    # warm-up), so the remaining time waits for a second.
    steps: int = 0
    # Cost and seconds measured since the images started (the rate of the
    # remaining time once there is enough of it: the text part says little
    # about the images).
    rate_cost: float = 0.0
    rate_seconds: float = 0.0
    log: str = ""
    error: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> JobState | None:
        d = json.loads(text)
        if not isinstance(d, dict) or d.get("version") != STATE_VERSION:
            return None
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def paths(self) -> list[str]:
        """Every path of the job: the current item and the queued ones."""
        return [*self.scope, *(p for item in self.queue for p in item)]


@dataclass(frozen=True)
class JobPaths:
    state: Path
    mutex: Path
    stop: Path
    log: Path
    final: Path  # the counts of a worker that stopped on request


def job_paths(index: Path) -> JobPaths:
    def side(suffix: str) -> Path:
        return index.with_name(index.name + suffix)

    return JobPaths(
        side("-job.json"), side("-jobmutex"), side("-job.stop"), side("-job.log"),
        side("-job.final"),
    )  # fmt: skip


@contextlib.contextmanager
def job_mutex(index: Path) -> Iterator[None]:
    """Hold the job mutex (milliseconds at a time; waits up to 5 s)."""
    path = job_paths(index).mutex
    _create_private(path)
    con = sqlite3.connect(str(path), timeout=5, isolation_level=None)
    try:
        con.execute("BEGIN EXCLUSIVE")
        yield
    finally:
        con.close()


def read_state(index: Path) -> JobState | None:
    path = job_paths(index).state
    for attempt in range(2):
        try:
            return JobState.from_json(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            if attempt == 0:
                time.sleep(0.05)  # caught a write in progress (Windows)
    return None


def write_state(index: Path, st: JobState) -> None:
    """Replace the state file atomically."""
    path = job_paths(index).state
    st.updated_at = time.time()
    tmp = path.with_name(path.name + ".tmp")
    _create_private(tmp)
    tmp.write_text(st.to_json(), encoding="utf-8")
    for _ in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # Windows: a reader has the file open
            time.sleep(0.05)
    with contextlib.suppress(OSError):  # the next update writes it again
        tmp.unlink()


def clear_state(index: Path) -> None:
    jp = job_paths(index)
    for p in (jp.state, jp.stop):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()


# ---------------------------------------------------------------- liveness
_children: dict[int, subprocess.Popen] = {}


def pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    child = _children.get(pid)
    if child is not None and child.poll() is not None:  # reaps a finished child
        return False
    if sys.platform == "win32":
        return _win_pid_running(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _win_pid_running(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    process_query_limited_information = 0x1000
    still_active = 259
    h = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not h:
        return ctypes.get_last_error() == 5  # access denied: it exists
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(h)


def process_start(pid: int) -> str | None:
    """When process `pid` started, as an opaque string that differs for a
    later process with the same PID; None when it cannot be found out."""
    if pid <= 0:
        return None
    try:
        if sys.platform == "win32":
            return _win_process_start(pid)
        if sys.platform.startswith("linux"):
            stat = Path(f"/proc/{pid}/stat").read_text()
            ticks = stat.rsplit(")", 1)[1].split()[19]  # field 22, starttime
            boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            return f"{boot}:{ticks}"
        out = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
            timeout=5, env={**os.environ, "LC_ALL": "C"},
        )  # fmt: skip
        return out.stdout.strip() or None
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _win_process_start(pid: int) -> str | None:
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        if not kernel32.GetProcessTimes(h, *(ctypes.byref(t) for t in times)):
            return None
        return str(times[0].value)  # creation time
    finally:
        kernel32.CloseHandle(h)


def same_process(st: JobState) -> bool | None:
    """Whether st.pid is still the process the state recorded: None when
    that cannot be checked (no start time recorded or readable)."""
    if not st.pid_start:
        return None
    now = process_start(st.pid)
    return None if now is None else now == st.pid_start


def lock_held(index: Path) -> bool:
    lock = acquire_writer_lock(index)
    if lock is None:
        return True
    lock.close()
    return False


def liveness(index: Path, st: JobState) -> str:
    """'alive' or 'stale'."""
    if st.phase == "failed":
        return "stale"
    if st.phase == "starting" and time.time() - st.started_at < STARTING_GRACE:
        return "alive" if st.pid <= 0 or pid_running(st.pid) else "stale"
    if not pid_running(st.pid) or same_process(st) is False:
        return "stale"
    return "alive" if lock_held(index) else "stale"


def check(index: Path, clean: bool = True) -> tuple[JobState | None, JobState | None]:
    """(the live job or None, a dead job that was just cleaned up or None).
    With clean=False a dead job is only ignored."""
    st = read_state(index)
    if st is None or liveness(index, st) == "alive":
        return st, None
    if not clean:
        return None, None
    with job_mutex(index):
        st = read_state(index)
        if st is None:
            return None, None
        if liveness(index, st) == "alive":
            return st, None
        clear_state(index)
    return None, st


def covers(st: JobState, scope: list[str]) -> bool:
    """Every path of `scope` is inside the job's item or a queued one."""
    return all(any(is_under(p, q) for q in st.paths) for p in scope)


def overlaps(paths: list[str], scope: list[str]) -> bool:
    return any(is_under(p, q) or is_under(q, p) for p in paths for q in scope)


def eta_seconds(st: JobState) -> float | None:
    """Seconds left for the current item, from the job's own measured speed;
    None while it is planned or before there is enough measurement."""
    if st.phase not in ("text", "images") or st.steps < ETA_MIN_STEPS:
        return None
    if st.rate_seconds >= RATE_MIN_SECONDS and st.rate_cost > 0:
        rate = st.rate_seconds / st.rate_cost
    elif st.embed_seconds >= RATE_MIN_SECONDS and st.cost_done > 0:
        rate = st.embed_seconds / st.cost_done
    else:
        return None
    return max(0.0, st.cost_total - st.cost_done) * rate


def meter(st: JobState, now: float | None = None, ascii: bool = False) -> str | None:
    """The job's progress in the format of the indexing progress bar:
    "Indexing 340/1363  25%|##5       | [12:03<35:10]"; None while it is
    still counting files."""
    from tqdm import tqdm

    from .indexer import BAR_FORMAT

    if st.phase not in ("text", "images"):
        return None
    eta = eta_seconds(st)
    remaining = "?" if eta is None else tqdm.format_interval(eta)
    total = st.cost_total if st.cost_total > 0 else 1.0
    return tqdm.format_meter(
        min(st.cost_done, total), total, max(0.0, (now or time.time()) - st.started_at),
        prefix=f"Indexing {st.files_done}/{st.files_total}",
        bar_format=BAR_FORMAT.replace("{remaining}", remaining), ascii=ascii,
    )  # fmt: skip


# ---------------------------------------------------------------- spawn
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _spawn_kwargs(platform: str, breakaway: bool = True) -> dict:
    if platform == "win32":
        # CREATE_NO_WINDOW, not DETACHED_PROCESS: in a venv or pipx install
        # sys.executable is a launcher that starts the real python.exe, which
        # would open a console window (or Terminal tab) of its own if the
        # launcher had no console. With a hidden console both share it.
        flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | BELOW_NORMAL_PRIORITY_CLASS
        return {"creationflags": flags | (CREATE_BREAKAWAY_FROM_JOB if breakaway else 0)}
    # Own session: no controlling terminal, no SIGHUP when it closes.
    return {"start_new_session": True}


def spawn(argv: list[str], log_path: Path, cwd: Path) -> int:
    """Start the worker detached, its output appended to log_path."""
    with contextlib.suppress(OSError):
        if log_path.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(log_path, log_path.with_name(log_path.name + ".1"))
    _create_private(log_path)
    with open(log_path, "ab") as log:
        kw = {
            "stdin": subprocess.DEVNULL,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "cwd": str(cwd),
            "close_fds": True,
        }
        try:
            p = subprocess.Popen(argv, **kw, **_spawn_kwargs(sys.platform))
        except OSError:
            if sys.platform != "win32":
                raise
            # The terminal's job object forbids breakaway: start it inside.
            p = subprocess.Popen(argv, **kw, **_spawn_kwargs(sys.platform, breakaway=False))
    _children[p.pid] = p
    return p.pid


def worker_argv(
    index: Path, paths: list[str], threads: int, options: list[str], job_id: str
) -> list[str]:
    return [
        *WORKER_PREFIX, "index", "--run-job", "--job-id", job_id, "--db", str(index),
        "-j", str(threads), *options, "--", *paths,
    ]  # fmt: skip


@dataclass
class Request:
    result: str  # started, attached, queued, busy
    state: JobState | None = None  # the job (for queued: the one it waits for)
    stale: JobState | None = None  # a dead job cleaned up on the way


def request(index: Path, paths: list[str], threads: int, options: list[str]) -> Request:
    """Start a job for `paths`, or add them to the running one. `paths` are
    canonical; an empty list means every indexed folder (only for a new job)."""
    index = Path(os.path.abspath(index))
    stale = None
    with job_mutex(index):
        st = read_state(index)
        if st is not None and liveness(index, st) == "stale":
            clear_state(index)
            stale, st = st, None
        if st is not None:
            if covers(st, paths):
                return Request("attached", st, stale)
            new = [p for p in paths if not any(is_under(p, q) for q in st.paths)]
            st.queue.append(new)
            write_state(index, st)
            return Request("queued", st, stale)
        if lock_held(index):
            return Request("busy", None, stale)
        jp = job_paths(index)
        st = JobState(
            job_id=uuid.uuid4().hex, started_at=time.time(), threads=threads, scope=paths,
            log=str(jp.log),
        )  # fmt: skip
        for old in (jp.stop, jp.final):
            with contextlib.suppress(FileNotFoundError):
                old.unlink()
        write_state(index, st)
    # Spawned outside the mutex, so the worker can take it at once.
    try:
        argv = worker_argv(index, paths, threads, options, st.job_id)
        pid = spawn(argv, job_paths(index).log, index.parent)
    except OSError:
        with job_mutex(index):
            clear_state(index)
        raise
    with job_mutex(index):
        cur = read_state(index)
        if cur is not None and cur.pid <= 0 and cur.job_id == st.job_id:
            cur.pid, cur.pid_start = pid, process_start(pid) or ""
            write_state(index, cur)
    st.pid = pid
    return Request("started", st, stale)


# ---------------------------------------------------------------- stop
def _take_final(index: Path) -> JobState | None:
    """The state a stopped worker left with its exact counts (then removed)."""
    path = job_paths(index).final
    try:
        st = JobState.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        st = None
    with contextlib.suppress(OSError):
        path.unlink()
    return st


@dataclass
class StopResult:
    result: str  # stopped, none, failed
    state: JobState | None = None
    stale: JobState | None = None


def _kill(pid: int) -> None:
    if sys.platform == "win32":
        os.kill(pid, signal.SIGTERM)  # TerminateProcess
    else:
        os.kill(pid, signal.SIGKILL)


def request_stop(index: Path) -> None:
    """Create the stop file the worker looks for. (_create_private creates
    files only on POSIX: on Windows the stop file was never made, and every
    stop waited STOP_WAIT and then killed the worker.)"""
    path = job_paths(index).stop
    _create_private(path)
    path.touch()


def stop(index: Path, timeout: float = STOP_WAIT) -> StopResult:
    st, stale = check(index)
    if st is None:
        return StopResult("none", None, stale)
    request_stop(index)
    last = st
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        cur = read_state(index)
        if cur is None:  # the worker cleaned up; give it a moment to exit
            end = time.monotonic() + KILL_WAIT
            while pid_running(last.pid) and time.monotonic() < end:
                time.sleep(0.1)
            return StopResult("stopped", _take_final(index) or last)
        last = cur
        if liveness(index, cur) == "stale":
            break
        time.sleep(0.2)
    cur = read_state(index)
    if cur is not None and liveness(index, cur) == "alive" and cur.pid > 0:
        # Kill only the worker of this job: never a process that merely has
        # its PID now, nor one whose start time cannot be checked.
        if cur.job_id != st.job_id or same_process(cur) is not True:
            return StopResult("failed", cur)
        try:
            _kill(cur.pid)
        except ProcessLookupError:
            pass
        except OSError:
            return StopResult("failed", cur)
        end = time.monotonic() + KILL_WAIT
        while pid_running(cur.pid) and time.monotonic() < end:
            time.sleep(0.1)
        if pid_running(cur.pid):
            return StopResult("failed", cur)
    with job_mutex(index):
        clear_state(index)
    return StopResult("stopped", last)


# ---------------------------------------------------------------- worker
def _detach_setup() -> None:
    """Lower the worker's priority and keep it alive when the terminal goes."""
    if hasattr(os, "nice"):
        with contextlib.suppress(OSError):
            os.nice(NICE)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def on_term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)


class _Worker:
    """The hooks run_index calls: progress into the state file, stop requests."""

    def __init__(self, index: Path, job_id: str = "") -> None:
        self.index = index
        self.job_id = job_id
        self.pid_start = process_start(os.getpid()) or ""
        self.stop_path = job_paths(index).stop
        self.last_write = 0.0
        self.phase = "planning"
        self.files_done = self.steps = 0
        self.cost_done = 0.0
        self.embed_seconds = 0.0
        self.rep = None  # the IndexReport of an interrupted run
        self.rate_start: tuple[float, float] | None = None  # (cost, seconds) when images began
        self.budget = 140  # image token budget of the run (set by cli._index_locked)

    def update(self, **fields) -> None:
        with job_mutex(self.index):
            st = read_state(self.index) or JobState(
                started_at=time.time(), log=str(job_paths(self.index).log)
            )
            st.pid, st.pid_start = os.getpid(), self.pid_start
            st.job_id = st.job_id or self.job_id
            for k, v in fields.items():
                setattr(st, k, v)
            write_state(self.index, st)
        self.last_write = time.monotonic()

    def should_stop(self) -> bool:
        return self.stop_path.exists()

    def on_plan(self, todo: list) -> list:
        from .indexer import estimate_cost

        if self.should_stop():
            raise KeyboardInterrupt
        budget = self.budget
        text = sum(e.kind != "image" for e, _ in todo)
        self.phase = "text" if text else "images"
        self.update(
            phase=self.phase,
            files_total=len(todo),
            text_total=text,
            cost_total=sum(estimate_cost(e, budget) for e, _ in todo),
        )
        return todo

    def on_progress(
        self, done: int, cost_done: float, seconds: float, kind: str, end: bool = True
    ) -> None:
        self.files_done, self.cost_done, self.embed_seconds = done, cost_done, seconds
        self.steps += end
        if self.should_stop():
            raise KeyboardInterrupt
        phase = "images" if kind == "image" else self.phase
        if phase == "images" and self.rate_start is None:
            self.rate_start = (cost_done, seconds)
        # Files finish in batches: save the state when a batch ends (not at
        # its first file, which would show one file of it until the next).
        age = time.monotonic() - self.last_write
        if phase != self.phase or age >= STATE_EVERY or (end and age >= STATE_MIN):
            self.phase = phase
            c0, s0 = self.rate_start or (cost_done, seconds)
            self.update(
                phase=phase, files_done=done, cost_done=cost_done, embed_seconds=seconds,
                rate_cost=cost_done - c0, rate_seconds=seconds - s0, steps=self.steps,
            )  # fmt: skip


def run_worker(args) -> int:
    """`anygrep index --run-job --db I -j N [options] -- PATH ...`."""
    from . import cli
    from .indexer import BG_CALIB_KEY, update_throughput
    from .store import Store

    index = Path(os.path.abspath(cli._index_path(args)))
    _detach_setup()
    w = _Worker(index, args.job_id or "")
    print(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} background index, PID {os.getpid()}")
    sys.stdout.flush()
    store = None
    try:
        store = Store(index, create=True)
        end = time.monotonic() + LOCK_WAIT
        while not store.try_write_lock(probe_db=False):
            if w.should_stop():
                raise KeyboardInterrupt
            if time.monotonic() > end:
                w.update(phase="failed", error="the index was busy")
                print("anygrep: the index was busy; background index not started", file=sys.stderr)
                return cli.EXIT_ERROR
            time.sleep(0.5)
        item = list(args.paths)
        while True:
            if w.should_stop():
                raise KeyboardInterrupt
            w.phase = "planning"
            w.cost_done = w.embed_seconds = 0.0
            w.files_done = w.steps = 0
            w.rate_start = None
            w.update(
                phase="planning", scope=item, files_total=0, files_done=0, text_total=0,
                cost_total=0.0, cost_done=0.0, embed_seconds=0.0, rate_cost=0.0,
                rate_seconds=0.0, steps=0, threads=args.threads,
            )  # fmt: skip
            args.paths = item
            rep = cli._index_locked(args, store, worker=w)
            code = cli._index_summary(args, rep)
            sys.stdout.flush()
            sys.stderr.flush()
            if code == cli.EXIT_INTERRUPTED:
                w.rep = rep
                raise KeyboardInterrupt
            with job_mutex(index):
                st = read_state(index)
                if st is not None and st.queue:
                    item = st.queue.pop(0)
                    write_state(index, st)
                    continue
                clear_state(index)
                store.close()  # releases the writer lock while the mutex is held
                store = None
            return 0
    except KeyboardInterrupt:
        r = w.rep
        if store is not None and r is not None:
            with contextlib.suppress(sqlite3.Error):
                update_throughput(store, r.embedded_cost, r.embed_seconds, BG_CALIB_KEY)
        with job_mutex(index):
            st = read_state(index)
            if st is not None and w.should_stop():
                # For `index --stop`, which reports the files done: exact
                # counts, not the last saved progress.
                if r is not None:
                    st.files_done = r.new + r.updated + r.skipped + r.errors
                with contextlib.suppress(OSError):
                    _create_private(job_paths(index).final)
                    job_paths(index).final.write_text(st.to_json(), encoding="utf-8")
            clear_state(index)
            if store is not None:
                store.close()
                store = None
        print("Background index stopped.", file=sys.stderr)
        return cli.EXIT_INTERRUPTED
    except BaseException as e:
        traceback.print_exc()
        msg = str(e) if not isinstance(e, cli._Exit) else "see the log"
        with contextlib.suppress(Exception):
            w.update(phase="failed", error=(msg or type(e).__name__)[:200])
        return cli.EXIT_ERROR
    finally:
        if store is not None:
            store.close()
        sys.stdout.flush()
        sys.stderr.flush()
