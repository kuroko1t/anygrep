"""Model cache coordination without network access or large model files."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from pathlib import Path

import pytest

from anygrep import model
from anygrep.config import ModelFile

CONTENT = b"small local model"
FILE = ModelFile("onnx/model.onnx", len(CONTENT), hashlib.sha256(CONTENT).hexdigest())


def _ensure_in_process(
    root: str, role: str, download_started, first_check_done, release_download, results
) -> None:
    os.environ["ANYGREP_MODEL_DIR"] = root
    os.environ.pop("ANYGREP_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    model.MODEL_FILES = (FILE,)

    if role == "first":

        def fake_download(_file, dest, _quiet):
            download_started.set()
            if not release_download.wait(10):
                raise RuntimeError("test did not release the download")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(CONTENT)

        model._download = fake_download
    else:
        original_missing_files = model.missing_files

        def observed_missing_files(vision, cache_root=None):
            result = original_missing_files(vision, cache_root)
            first_check_done.set()
            return result

        def unexpected_download(_file, _dest, _quiet):
            raise AssertionError("second process downloaded an already completed file")

        model.missing_files = observed_missing_files
        model._download = unexpected_download

    try:
        results.put((role, str(model.ensure_model(vision=False, quiet=True))))
    except Exception as exc:
        results.put((role, repr(exc)))


def test_second_process_rechecks_after_download_lock(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    download_started = ctx.Event()
    first_check_done = ctx.Event()
    release_download = ctx.Event()
    results = ctx.Queue()
    root = tmp_path / "cache"
    first = ctx.Process(
        target=_ensure_in_process,
        args=(str(root), "first", download_started, first_check_done, release_download, results),
    )
    second = ctx.Process(
        target=_ensure_in_process,
        args=(str(root), "second", download_started, first_check_done, release_download, results),
    )
    first.start()
    try:
        assert download_started.wait(10), "first process never reached the download"
        second.start()
        assert first_check_done.wait(10), "second process never checked missing files"
    finally:
        release_download.set()
        first.join(15)
        if second.pid is not None:
            second.join(15)

    assert first.exitcode == second.exitcode == 0
    assert {results.get(timeout=2) for _ in range(2)} == {
        ("first", str(root)),
        ("second", str(root)),
    }
    assert (root / FILE.path).read_bytes() == CONTENT
    assert model._marker(root / FILE.path).is_file()
    assert (root / ".model-download.lock").is_file()


@pytest.mark.parametrize("invalid", ["[]", "null", "42"])
def test_malformed_marker_is_reverified(tmp_path: Path, monkeypatch, invalid: str) -> None:
    monkeypatch.setattr(model, "MODEL_FILES", (FILE,))
    monkeypatch.setenv("ANYGREP_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ANYGREP_OFFLINE", "1")
    dest = tmp_path / FILE.path
    dest.parent.mkdir(parents=True)
    dest.write_bytes(CONTENT)
    marker = model._marker(dest)
    marker.write_text(invalid)

    assert model.missing_files(vision=False) == [FILE]
    assert model.ensure_model(vision=False, quiet=True) == tmp_path
    assert model.missing_files(vision=False) == []
    assert list(dest.parent.glob(marker.name + ".*.tmp")) == []


def test_verified_cache_does_not_create_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model, "MODEL_FILES", (FILE,))
    monkeypatch.setenv("ANYGREP_MODEL_DIR", str(tmp_path))
    dest = tmp_path / FILE.path
    dest.parent.mkdir(parents=True)
    dest.write_bytes(CONTENT)
    assert model._verify(dest, FILE)

    assert model.ensure_model(vision=False, quiet=True) == tmp_path
    assert not (tmp_path / ".model-download.lock").exists()


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_offline_read_only_cache_reports_offline_message(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model, "MODEL_FILES", (FILE,))
    root = tmp_path / "ro"
    root.mkdir()
    root.chmod(0o555)
    monkeypatch.setenv("ANYGREP_MODEL_DIR", str(root))
    monkeypatch.setenv("ANYGREP_OFFLINE", "1")
    try:
        with pytest.raises(model.ModelUnavailableError, match="download these files"):
            model.ensure_model(vision=False, quiet=True)
    finally:
        root.chmod(0o755)
    assert not (root / ".model-download.lock").exists()


def test_lock_failure_falls_back_to_unlocked_download(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model, "MODEL_FILES", (FILE,))
    monkeypatch.setenv("ANYGREP_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("ANYGREP_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def no_lock(*_args, **_kwargs):
        raise model.sqlite3.OperationalError("disk I/O error")

    def fake_download(_file, dest, _quiet):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(CONTENT)

    monkeypatch.setattr(model.sqlite3, "connect", no_lock)
    monkeypatch.setattr(model, "_download", fake_download)
    assert model.ensure_model(vision=False, quiet=True) == tmp_path
    assert model.missing_files(vision=False) == []


def _hold_lock(root: Path):
    import sqlite3

    root.mkdir(parents=True, exist_ok=True)
    holder = sqlite3.connect(
        str(root.resolve() / ".model-download.lock"), isolation_level=None, check_same_thread=False
    )
    holder.execute("BEGIN EXCLUSIVE")
    return holder


def test_waiting_for_lock_is_reported_and_times_out(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(model, "LOCK_WAIT", 0.3)
    monkeypatch.setattr(model, "LOCK_POLL", 0.05)
    holder = _hold_lock(tmp_path)
    try:
        with (
            pytest.raises(model.ModelUnavailableError, match="busy"),
            model._model_lock(tmp_path, quiet=False),
        ):
            pass
    finally:
        holder.close()
    err = capsys.readouterr().err
    assert err.count("Waiting for another anygrep process") == 1


def test_waiter_gets_lock_after_release(tmp_path: Path, monkeypatch, capsys) -> None:
    import threading

    monkeypatch.setattr(model, "LOCK_WAIT", 10.0)
    monkeypatch.setattr(model, "LOCK_POLL", 0.05)
    holder = _hold_lock(tmp_path)
    timer = threading.Timer(0.2, holder.close)
    timer.start()
    try:
        with model._model_lock(tmp_path, quiet=True):
            pass
    finally:
        timer.join()
    assert capsys.readouterr().err == ""
