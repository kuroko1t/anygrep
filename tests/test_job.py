"""Background indexing: state file, liveness, requests, ETA and the worker
(in-process, with the fake model)."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from PIL import Image

from anygrep import cli, indexer, job
from anygrep.indexer import index_order
from anygrep.store import Store, acquire_writer_lock
from anygrep.walker import FileEntry

from .conftest import sync_spawn


def db_path() -> Path:
    return cli.default_index_path()


# 1 ---------------------------------------------------------------- ordering
def test_index_order(tmp_path: Path) -> None:
    def e(name: str, kind: str, mtime: int) -> tuple[FileEntry, bool]:
        return FileEntry(tmp_path / name, kind, 10, mtime), True

    todo = [
        e("b.png", "image", 1), e("z.md", "text", 9), e("a.png", "image", 5),
        e("c.png", "image", 5), e("doc.pdf", "pdf", 1), e("a.md", "text", 1),
    ]  # fmt: skip
    got = [t[0].path.name for t in index_order(todo)]
    assert got == ["a.md", "z.md", "doc.pdf", "a.png", "c.png", "b.png"]


# 2 ---------------------------------------------------------------- state
def test_state_round_trip(tmp_path: Path, monkeypatch) -> None:
    index = tmp_path / "index.sqlite"
    assert job.read_state(index) is None
    st = job.JobState(pid=12, scope=["/a"], queue=[["/b"]], phase="text", files_total=3)
    job.write_state(index, st)
    got = job.read_state(index)
    assert got == st and got.paths == ["/a", "/b"]
    assert not list(tmp_path.glob("*.tmp"))
    jp = job.job_paths(index)
    jp.state.write_text("{not json", encoding="utf-8")
    assert job.read_state(index) is None
    jp.state.write_text('{"version": 2, "pid": 1}', encoding="utf-8")
    assert job.read_state(index) is None
    # Windows: os.replace fails while a reader has the file open.
    fails = [PermissionError, PermissionError]
    real = os.replace

    def flaky(a, b):
        if fails:
            raise fails.pop()()
        real(a, b)

    monkeypatch.setattr(job.os, "replace", flaky)
    job.write_state(index, st)
    assert job.read_state(index) == st and not fails


# 3 ---------------------------------------------------------------- liveness
def test_liveness(tmp_path: Path, monkeypatch) -> None:
    index = tmp_path / "index.sqlite"
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True)  # fmt: skip
    st = job.JobState(pid=int(dead.stdout), phase="images", started_at=time.time())
    assert job.liveness(index, st) == "stale"
    st.pid = os.getpid()
    assert job.liveness(index, st) == "stale"  # the writer lock is free
    lock = acquire_writer_lock(index)
    try:
        assert job.liveness(index, st) == "alive"
        st.phase = "failed"
        assert job.liveness(index, st) == "stale"
    finally:
        lock.close()
    starting = job.JobState(pid=0, phase="starting", started_at=time.time())
    assert job.liveness(index, starting) == "alive"  # still importing
    later = time.time() + job.STARTING_GRACE + 1
    monkeypatch.setattr(job.time, "time", lambda: later)
    assert job.liveness(index, starting) == "stale"


# 4 ---------------------------------------------------------------- requests
@pytest.fixture
def fake_spawn(monkeypatch):
    calls: list[list[str]] = []

    def spawn(argv, log, cwd):
        calls.append(argv)
        return 999_999_999

    monkeypatch.setattr(job, "spawn", spawn)
    return calls


def test_request_rows(tmp_path: Path, fake_spawn) -> None:
    index = tmp_path / "index.sqlite"
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    r = job.request(index, [a], 2, ["--no-images"])
    assert r.result == "started" and r.state.pid == 999_999_999
    argv = fake_spawn[0]
    st = job.read_state(index)
    assert argv[len(job.WORKER_PREFIX) :] == [
        "index", "--run-job", "--job-id", st.job_id, "--db", str(index), "-j", "2",
        "--no-images", "--", a,
    ]  # fmt: skip
    assert st.phase == "starting" and st.pid == 999_999_999 and st.scope == [a]
    assert len(st.job_id) == 32
    # A running job (this process holds the lock and is the PID).
    st.pid, st.phase = os.getpid(), "images"
    job.write_state(index, st)
    lock = acquire_writer_lock(index)
    try:
        assert job.request(index, [a + os.sep + "sub"], 2, []).result == "attached"
        r = job.request(index, [b, a], 2, [])
        assert r.result == "queued" and job.read_state(index).queue == [[b]]
        assert job.request(index, [b], 2, []).result == "attached"
        job.clear_state(index)
        assert job.request(index, [b], 2, []).result == "busy"  # a foreground writer
    finally:
        lock.close()
    assert len(fake_spawn) == 1
    # A dead job is cleaned up on the way and reported.
    st.pid = 999_999_998
    job.write_state(index, st)
    r = job.request(index, [b], 2, [])
    assert r.result == "started" and r.stale is not None and r.stale.scope == [a]


# 5 ---------------------------------------------------------------- ETA, lines
def test_eta_and_progress_lines(tmp_path: Path, monkeypatch) -> None:
    now = 1_800_000_000.0
    monkeypatch.setattr(job.time, "time", lambda: now)
    st = job.JobState(scope=[str(tmp_path)], phase="planning", started_at=now - 723)
    assert job.eta_seconds(st) is None and job.meter(st) is None
    where = cli._tilde(str(tmp_path))
    assert cli._job_progress(st, []) == (
        f"background index: counting files under {where} (anygrep status)"
    )
    # No up-front estimate: "?" until the job has measured its own speed.
    st.phase, st.files_total, st.files_done = "text", 1363, 3
    st.cost_total, st.cost_done, st.embed_seconds = 4000.0, 10.0, 5.0
    assert job.eta_seconds(st) is None
    assert job.meter(st, ascii=True) == "Indexing 3/1363   0%|          | [12:03<?]"
    st.phase, st.files_done = "images", 340
    st.cost_done, st.embed_seconds = 1000.0, 3000.0
    st.steps = 1  # one batch: its speed says little yet
    assert job.eta_seconds(st) is None
    st.steps = 2
    assert job.eta_seconds(st) == 9000.0  # the overall measured rate
    st.rate_cost, st.rate_seconds = 100.0, 200.0
    assert job.eta_seconds(st) == 6000.0  # the measured rate of the images
    assert cli._job_progress(st, [str(tmp_path / "sub")]) == (
        "background index: Indexing 340/1363  25%|\u2588\u2588\u258c       | [12:03<1:40:00] "
        "(anygrep status)"
    )
    other = str(tmp_path.parent / "elsewhere")
    assert cli._job_progress(st, [other]) == (
        f"a background index of {where} is running, so {cli._tilde(other)} was not checked "
        f"for new files; add it with: anygrep index --background {cli._tilde(other)}"
    )
    st.queue = [[other]]
    assert cli._job_progress(st, [other]) == (
        f"{cli._tilde(other)} is queued for background indexing after {where} (anygrep status)"
    )
    monkeypatch.setattr(cli, "_DB_HINT", " --db /x.sqlite")
    assert cli._job_progress(st, []).endswith("(anygrep status --db /x.sqlite)")


# 6 ---------------------------------------------------------------- worker
def make_corpus(root: Path, images: int = 6) -> Path:
    root.mkdir(parents=True)
    (root / "a.md").write_text("bread and butter\n", encoding="utf-8")
    (root / "b.md").write_text("bicycle chain\n", encoding="utf-8")
    for i in range(images):
        Image.new("RGB", (64, 64), (200, i * 30, 10)).save(root / f"p{i}.png")
        os.utime(root / f"p{i}.png", (1_700_000_000 + i, 1_700_000_000 + i))
    return root


def test_worker_in_process(env, tmp_path: Path, monkeypatch, capsys) -> None:
    corpus = make_corpus(tmp_path / "c")
    other = make_corpus(tmp_path / "o", images=1)
    monkeypatch.setattr(indexer, "CALIB_MIN_SECONDS", 0.0)
    monkeypatch.setattr(job, "STATE_EVERY", 0.0)
    phases: list[str] = []
    real_write = job.write_state

    def write(index, st):
        if not phases or phases[-1] != st.phase:
            phases.append(st.phase)
        real_write(index, st)

    monkeypatch.setattr(job, "write_state", write)
    seen: list[tuple[int, int]] = []
    real_embed = env.embed_image

    def embed(im):
        n = env.calls["image"]
        if n == 2:  # progressive commits: another reader sees the rows so far
            st = Store(db_path(), create=False, readonly=True)
            seen.append((n, st.counts().get("image", {}).get("ok", 0)))
            st.close()
            with job.job_mutex(db_path()):
                s = job.read_state(db_path())
                s.queue.append([str(other)])
                real_write(db_path(), s)
        return real_embed(im)

    monkeypatch.setattr(env, "embed_image", embed)
    monkeypatch.setattr(indexer, "COMMIT_EVERY_SEC", 0.0)
    del sync_spawn.codes[:]
    assert cli.main(["index", "--background", str(corpus)]) == 0
    assert sync_spawn.codes == [0]
    assert phases[:3] == ["starting", "planning", "text"] and "images" in phases
    assert seen == [(2, 2)]  # the two newest images were committed
    assert not job.job_paths(db_path()).state.exists()
    st = Store(db_path(), create=False)
    try:
        assert st.counts()["image"]["ok"] == 7 and st.counts()["text"]["ok"] == 4
        assert st.get_meta(indexer.BG_CALIB_KEY) is not None
        assert st.get_meta(indexer.CALIB_KEY) is None  # foreground speed untouched
    finally:
        st.close()
    assert "in the background" in capsys.readouterr().err


def test_worker_stop_and_resume(env, tmp_path: Path, monkeypatch, capsys) -> None:
    corpus = make_corpus(tmp_path / "c", images=8)
    real_embed = env.embed_image
    monkeypatch.setattr(indexer, "COMMIT_EVERY_SEC", 0.0)

    def embed(im):
        if env.calls["image"] == 3:
            job.job_paths(db_path()).stop.touch()
        return real_embed(im)

    monkeypatch.setattr(env, "embed_image", embed)
    del sync_spawn.codes[:]
    assert cli.main(["index", "--background", str(corpus)]) == 0
    assert sync_spawn.codes == [130]
    jp = job.job_paths(db_path())
    assert not jp.state.exists() and not jp.stop.exists()
    st = Store(db_path(), create=False)
    assert 0 < st.counts()["image"]["ok"] < 8
    st.close()
    monkeypatch.setattr(env, "embed_image", real_embed)
    before = env.calls["image"]
    assert cli.main(["index", "--background", str(corpus)]) == 0
    assert sync_spawn.codes == [130, 0]
    st = Store(db_path(), create=False)
    assert st.counts()["image"]["ok"] == 8
    st.close()
    assert env.calls["image"] - before == 8 - 4  # only the remaining ones
    assert "Stopped after" in jp.log.read_text(encoding="utf-8")
    capsys.readouterr()


def test_worker_failure_is_reported_as_stale(env, tmp_path: Path, monkeypatch, capsys) -> None:
    corpus = make_corpus(tmp_path / "c", images=1)

    monkeypatch.setattr(job._Worker, "on_plan", lambda self, todo: (_ for _ in ()).throw(
        OSError("disk full")))  # fmt: skip
    del sync_spawn.codes[:]
    assert cli.main(["index", "--background", str(corpus)]) == 0
    assert sync_spawn.codes == [2]
    st = job.read_state(db_path())
    assert st.phase == "failed" and st.error == "disk full"
    capsys.readouterr()
    assert cli.main(["status"]) == 0
    err = capsys.readouterr().err
    assert "stopped before it finished; resume with: anygrep index --background" in err
    assert "  failed: disk full; log: " in err
    assert job.read_state(db_path()) is None


# 9 ---------------------------------------------------------------- spawn
def test_spawn_kwargs_and_breakaway_fallback(tmp_path: Path, monkeypatch) -> None:
    win = job._spawn_kwargs("win32")["creationflags"]
    assert win & job.CREATE_NO_WINDOW and win & job.CREATE_NEW_PROCESS_GROUP
    assert not win & 0x00000008  # DETACHED_PROCESS opens a window via the venv launcher
    assert win & job.BELOW_NORMAL_PRIORITY_CLASS and win & job.CREATE_BREAKAWAY_FROM_JOB
    assert not job._spawn_kwargs("win32", breakaway=False)["creationflags"] & (
        job.CREATE_BREAKAWAY_FROM_JOB
    )
    assert job._spawn_kwargs("linux") == {"start_new_session": True}
    assert job._spawn_kwargs("darwin") == {"start_new_session": True}
    calls = []

    class FakePopen:
        pid = 4242

        def __init__(self, argv, **kw):
            calls.append(kw.get("creationflags"))
            if len(calls) == 1:
                raise PermissionError("access denied")

        def poll(self):
            return None

    monkeypatch.setattr(job.sys, "platform", "win32")
    monkeypatch.setattr(job.subprocess, "Popen", FakePopen)
    assert job.spawn(["x"], tmp_path / "log", tmp_path) == 4242
    assert calls[0] & job.CREATE_BREAKAWAY_FROM_JOB
    assert not calls[1] & job.CREATE_BREAKAWAY_FROM_JOB
    job._children.pop(4242, None)


# 10 --------------------------------------------------------------- review fixes
def test_progress_is_saved_when_a_batch_ends(tmp_path: Path, monkeypatch) -> None:
    """Files of a batch finish together: the state shows the whole batch,
    not its first file until the next batch."""
    index = tmp_path / "index.sqlite"
    monkeypatch.setattr(job, "STATE_EVERY", 3600.0)
    monkeypatch.setattr(job, "STATE_MIN", 0.0)
    w = job._Worker(index)
    w.update(phase="text")
    w.phase = "text"
    done = 0
    for _ in range(4):
        for i in range(8):
            done += 1
            w.on_progress(done, float(done), 1.0, "text", i == 7)
        assert job.read_state(index).files_done == done


def test_run_index_reports_a_text_batch_as_one_step(tmp_path: Path, fake_embedder) -> None:
    from anygrep.indexer import IndexOptions, run_index

    d = tmp_path / "notes"
    d.mkdir()
    for i in range(5):
        (d / f"{i}.md").write_text(f"note number {i} about bread\n", encoding="utf-8")
    calls = []
    opts = IndexOptions(quiet=True, on_progress=lambda *a: calls.append((a[0], a[4])))
    store = Store(tmp_path / "idx.sqlite")
    try:
        run_index(store, [d], opts, lambda vision: fake_embedder)
    finally:
        store.close()
    assert calls == [(1, False), (2, False), (3, False), (4, False), (5, True)]


def test_stop_is_seen_between_model_runs(monkeypatch) -> None:
    from anygrep import embedder as E

    emb = E.OnnxEmbedder.__new__(E.OnnxEmbedder)
    emb._empty = None
    runs = []
    monkeypatch.setattr(
        emb, "_run_lm", lambda ids, mask, feats: runs.append(len(ids)) or [[0.0]] * len(ids)
    )
    seqs = [[1] * 3000 for _ in range(4)]  # one model run each (4096 tokens)
    emb.should_stop = None
    emb._embed_ids(seqs)
    assert runs == [1, 1, 1, 1]
    runs.clear()
    emb.should_stop = lambda: len(runs) >= 2
    with pytest.raises(KeyboardInterrupt):
        emb._embed_ids(seqs)
    assert runs == [1, 1]


@pytest.mark.parametrize("command", [["index", "x"], ["remove", "x"], ["reset", "-y"]])
def test_foreground_commands_wait_for_a_starting_job(env, tmp_path: Path, capsys, command) -> None:
    """A job whose worker has not taken the writer lock yet keeps index,
    remove and reset out, as a running one does."""
    corpus = make_corpus(tmp_path / "c", images=0)
    assert cli.main(["index", "-q", str(corpus)]) == 0
    command = [str(corpus) if a == "x" else a for a in command]
    job.write_state(db_path(), job.JobState(started_at=time.time(), scope=[str(corpus)]))
    capsys.readouterr()
    assert cli.main(command) == 2
    err = capsys.readouterr().err
    assert "a background index is running" in err
    assert db_path().exists() and not job.lock_held(db_path())
    job.clear_state(db_path())


def test_worker_waiting_for_the_lock_sees_a_stop(env, tmp_path: Path, monkeypatch) -> None:
    corpus = make_corpus(tmp_path / "c", images=0)
    assert cli.main(["index", "-q", str(corpus)]) == 0
    lock = acquire_writer_lock(db_path())
    try:
        job.job_paths(db_path()).stop.touch()
        args = cli.parse_args(["index", "--run-job", "--db", str(db_path()), "--", str(corpus)])
        args.threads = 1
        t0 = time.monotonic()
        assert job.run_worker(args) == cli.EXIT_INTERRUPTED
        assert time.monotonic() - t0 < 5
    finally:
        lock.close()


def test_request_stop_creates_the_file_on_every_platform(tmp_path: Path, monkeypatch) -> None:
    index = tmp_path / "index.sqlite"
    # What _create_private does on Windows: the folder only, no file.
    monkeypatch.setattr(job, "_create_private", lambda p: p.parent.mkdir(exist_ok=True))
    job.request_stop(index)
    assert job.job_paths(index).stop.exists()
    assert job._Worker(index).should_stop()


# 11 --------------------------------------------------------------- stop: whose PID
def test_process_start_tells_processes_apart() -> None:
    me = job.process_start(os.getpid())
    if me is None:
        pytest.skip("process start times are not available here")
    assert job.process_start(os.getpid()) == me
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert job.process_start(other.pid) not in (None, me)
    finally:
        other.kill()
        other.wait()
    st = job.JobState(pid=os.getpid(), pid_start=me)
    assert job.same_process(st) is True
    st.pid_start = "another process"
    assert job.same_process(st) is False
    st.pid_start = ""
    assert job.same_process(st) is None


@pytest.fixture
def reused_pid(tmp_path: Path, monkeypatch):
    """A job state whose PID 4242 now belongs to another, running process
    (the worker died hard and the PID was reused), while something else
    holds the writer lock. Kills are recorded, never sent."""
    index = tmp_path / "index.sqlite"
    st = job.JobState(job_id="a" * 32, pid=4242, pid_start="worker", phase="text",
                      started_at=time.time() - 100, scope=[str(tmp_path)])  # fmt: skip
    job.write_state(index, st)
    starts = {4242: "another process"}
    killed: list[int] = []
    monkeypatch.setattr(job, "pid_running", lambda pid: pid == 4242)
    monkeypatch.setattr(job, "process_start", lambda pid: starts.get(pid))
    monkeypatch.setattr(job, "_kill", killed.append)
    lock = acquire_writer_lock(index)
    yield index, starts, killed
    lock.close()


def test_stop_never_kills_a_reused_pid(reused_pid) -> None:
    index, _, killed = reused_pid
    r = job.stop(index, timeout=0.5)
    assert killed == []
    assert r.result == "none" and r.stale is not None and r.stale.pid == 4242
    assert not job.job_paths(index).state.exists()


def test_stop_does_not_kill_when_ownership_is_unknown(reused_pid) -> None:
    index, starts, killed = reused_pid
    del starts[4242]  # the start time cannot be read
    r = job.stop(index, timeout=0.5)
    assert killed == [] and r.result == "failed"
    st = job.read_state(index)  # recorded without a start time (older state)
    st.pid_start = ""
    job.write_state(index, st)
    starts[4242] = "worker"
    assert job.stop(index, timeout=0.5).result == "failed" and killed == []


def test_stop_kills_its_own_worker(reused_pid, monkeypatch) -> None:
    index, starts, killed = reused_pid
    starts[4242] = "worker"
    monkeypatch.setattr(job, "pid_running", lambda pid: pid == 4242 and not killed)
    r = job.stop(index, timeout=0.5)
    assert killed == [4242] and r.result == "stopped"
    assert not job.job_paths(index).state.exists()


def test_stop_returns_after_the_worker_cleaned_up(tmp_path: Path, monkeypatch) -> None:
    """`index --stop` leaves no stop file, state or final counts behind, even
    when the worker sees the stop file and cleans up at once (the stop file
    used to be made in two steps, and the second one could make it again)."""
    import threading

    index = tmp_path / "index.sqlite"
    jp = job.job_paths(index)
    me = os.getpid()
    st = job.JobState(job_id="b" * 32, pid=me, pid_start=job.process_start(me) or "",
                      phase="images", started_at=time.time(), scope=[str(tmp_path)],
                      files_total=10, files_done=3)  # fmt: skip
    job.write_state(index, st)
    holding = threading.Event()
    real_create = job._create_private

    def slow_create(p: Path) -> None:
        real_create(p)
        if p == jp.stop:
            time.sleep(0.3)  # the stopping process is slow to go on

    monkeypatch.setattr(job, "_create_private", slow_create)
    monkeypatch.setattr(job, "KILL_WAIT", 0.3)  # this "worker" never exits

    def worker() -> None:  # what run_worker does when it sees the stop file
        lock = acquire_writer_lock(index)
        holding.set()
        end = time.monotonic() + 10
        while not jp.stop.exists() and time.monotonic() < end:
            time.sleep(0.01)
        with job.job_mutex(index):
            cur = job.read_state(index)
            cur.files_done = 4
            jp.final.write_text(cur.to_json(), encoding="utf-8")
            job.clear_state(index)
            lock.close()

    t = threading.Thread(target=worker)
    t.start()
    try:
        assert holding.wait(10)
        r = job.stop(index, timeout=5)
    finally:
        t.join(10)
    assert r.result == "stopped" and r.state.files_done == 4
    assert not jp.state.exists() and not jp.stop.exists() and not jp.final.exists()
    assert not job.lock_held(index)


# 12 --------------------------------------------------------------- stop while starting
def _worker_args(index: Path, job_id: str, folder: Path):
    args = cli.parse_args(
        ["index", "--run-job", "--job-id", job_id, "--db", str(index), "--", str(folder)]
    )
    args.threads = 1
    return args


def test_stop_during_a_slow_spawn_cancels_the_job(tmp_path: Path, monkeypatch) -> None:
    import threading

    index = tmp_path / "index.sqlite"
    release = threading.Event()
    started = threading.Event()

    def slow_spawn(argv, log, cwd):
        started.set()
        assert release.wait(10)
        return 999_999_997  # a fake PID: not a child of this process, never signalled

    monkeypatch.setattr(job, "spawn", slow_spawn)
    monkeypatch.setattr(job, "_kill", lambda pid: pytest.fail("nothing may be killed"))
    out: list[job.Request] = []
    t = threading.Thread(target=lambda: out.append(job.request(index, [str(tmp_path)], 1, [])))
    t.start()
    try:
        assert started.wait(10)
        job_id = job.read_state(index).job_id
        r = job.stop(index, timeout=0.3)
        assert r.result == "stopped"
        st = job.read_state(index)
        assert st.phase == "cancelled" and st.job_id == job_id
        assert job.job_paths(index).stop.exists()
        # A worker that gets going before the spawn returns exits at once,
        # without opening the index or touching the state.
        assert job.run_worker(_worker_args(index, job_id, tmp_path)) == cli.EXIT_INTERRUPTED
        assert not index.exists() and job.read_state(index).phase == "cancelled"
    finally:
        release.set()
        t.join(10)
    assert out[0].result == "cancelled"
    assert job.read_state(index) is None and not job.job_paths(index).stop.exists()
    assert job.run_worker(_worker_args(index, job_id, tmp_path)) == cli.EXIT_INTERRUPTED
    assert not index.exists()
    # Nothing is left that reads as a lost job.
    assert job.check(index) == (None, None)


def test_a_late_worker_does_not_take_over_a_newer_job(tmp_path: Path) -> None:
    index = tmp_path / "index.sqlite"
    newer = job.JobState(job_id="b" * 32, pid=4242, pid_start="b", phase="text",
                         started_at=time.time(), scope=[str(tmp_path)])  # fmt: skip
    job.write_state(index, newer)
    before = job.job_paths(index).state.read_text(encoding="utf-8")
    late = job._Worker(index, "a" * 32)
    with pytest.raises(KeyboardInterrupt):
        late.update(phase="planning")
    assert late.should_stop()
    with contextlib.suppress(KeyboardInterrupt):
        late.update(phase="failed", error="the index was busy")
    assert job.job_paths(index).state.read_text(encoding="utf-8") == before
    job.request_stop(index)  # the newer job's stop request stays for it
    assert job.run_worker(_worker_args(index, "a" * 32, tmp_path)) == cli.EXIT_INTERRUPTED
    assert job.job_paths(index).state.read_text(encoding="utf-8") == before
    assert job.job_paths(index).stop.exists() and not index.exists()
    # Its own worker still updates it.
    job._Worker(index, "b" * 32).update(files_done=3)
    st = job.read_state(index)
    assert st.files_done == 3 and st.pid == os.getpid()


def test_process_start_does_not_depend_on_the_time_zone(monkeypatch) -> None:
    import shutil

    if shutil.which("ps") is None:
        pytest.skip("no ps here")
    monkeypatch.setattr(job.sys, "platform", "darwin")  # the `ps -o lstart=` path
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    tokyo = job.process_start(os.getpid())
    if tokyo is None:
        pytest.skip("ps -o lstart is not available here")
    monkeypatch.setenv("TZ", "UTC")
    assert job.process_start(os.getpid()) == tokyo
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert job.process_start(os.getpid()) == tokyo


@pytest.mark.skipif(sys.platform == "win32", reason="uses a POSIX sleep child")
def test_stop_ignores_a_final_file_left_by_another_job(tmp_path):
    # A final file from an earlier job must not be reported as this job's
    # counts when the live worker has to be killed.
    from anygrep.store import acquire_writer_lock

    index = tmp_path / "idx.db"
    lock = acquire_writer_lock(index)
    child = subprocess.Popen(["sleep", "60"])
    job._children[child.pid] = child
    try:
        old = job.JobState(started_at=time.time(), log="x", job_id="A", files_done=999)
        job.job_paths(index).final.write_text(old.to_json())
        cur = job.JobState(
            started_at=time.time(),
            log="x",
            job_id="B",
            pid=child.pid,
            pid_start=job.process_start(child.pid) or "",
            phase="text",
            files_done=3,
        )
        job.write_state(index, cur)
        res = job.stop(index, timeout=1)
        assert res.result == "stopped"
        assert res.state is not None and res.state.job_id == "B"
        assert not job.job_paths(index).final.exists()
    finally:
        with contextlib.suppress(OSError):
            child.kill()
        child.wait()
        lock.close()
