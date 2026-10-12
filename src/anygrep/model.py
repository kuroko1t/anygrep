"""Model file download and verification.

Only the files the tool needs are fetched (tokenizer, int8 text model and,
when images are used, the int8 vision encoder), pinned to one revision and
checked against known SHA-256 hashes. This is the only network access the
tool performs.
"""

from __future__ import annotations

import http.client
import json
import os
import sqlite3
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .config import MODEL_FILES, MODEL_REPO, MODEL_REVISION, ModelFile, model_dir, offline
from .errors import ModelUnavailableError
from .paths import sha256_file

CHUNK = 1 << 20
RETRY_DELAY = 1.0  # seconds, doubled per attempt
LOCK_WAIT = 900.0  # a second process may need to download several large files


def _endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def file_url(f: ModelFile) -> str:
    return f"{_endpoint()}/{MODEL_REPO}/resolve/{MODEL_REVISION}/{f.path}"


def needed_files(vision: bool) -> list[ModelFile]:
    return [f for f in MODEL_FILES if vision or not f.vision]


def _marker(path: Path) -> Path:
    return path.with_name(path.name + ".verified")


def _is_verified(path: Path, f: ModelFile) -> bool:
    m = _marker(path)
    if not path.is_file() or not m.is_file():
        return False
    try:
        info = json.loads(m.read_text())
        if not isinstance(info, dict):
            return False
        st = path.stat()
        return (
            info.get("sha256") == f.sha256
            and info.get("size") == st.st_size == f.size
            and info.get("mtime_ns") == st.st_mtime_ns
        )
    except (OSError, ValueError):
        return False


def _verify(path: Path, f: ModelFile) -> bool:
    if not path.is_file() or path.stat().st_size != f.size:
        return False
    if sha256_file(path) != f.sha256:
        return False
    st = path.stat()
    marker = _marker(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=marker.parent,
            prefix=marker.name + ".",
            suffix=".tmp",
            delete=False,
        ) as out:
            temporary = out.name
            json.dump({"sha256": f.sha256, "size": st.st_size, "mtime_ns": st.st_mtime_ns}, out)
        os.replace(temporary, marker)
    except OSError:
        # A read-only cache, or (Windows) a marker another process has open:
        # the file is verified either way, it is just hashed again next time.
        pass
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return True


LOCK_POLL = 0.5  # seconds between attempts; a Python sleep keeps Ctrl-C working


def _acquire_model_lock(root: Path, quiet: bool) -> sqlite3.Connection | None:
    """Take the cache's cross-process lock, or return None when locking is not
    possible here (read-only or unusual filesystem): the lock only prevents two
    processes downloading the same files, so it is best-effort."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock_file = root.resolve() / ".model-download.lock"
        lock = sqlite3.connect(str(lock_file), timeout=0, isolation_level=None)
    except (OSError, sqlite3.Error):
        return None
    deadline = time.monotonic() + LOCK_WAIT
    waiting = False
    try:
        while True:
            try:
                lock.execute("BEGIN EXCLUSIVE")
                return lock
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" not in msg and "busy" not in msg:
                    lock.close()
                    return None
            if time.monotonic() >= deadline:
                lock.close()
                raise ModelUnavailableError(
                    f"Model cache at {root} is busy; another process may be downloading "
                    "the model. Please retry."
                )
            if not waiting and not quiet:
                print(
                    f"Waiting for another anygrep process to finish downloading the model ({root})...",
                    file=sys.stderr,
                )
            waiting = True
            time.sleep(LOCK_POLL)
    except BaseException:
        lock.close()
        raise


@contextmanager
def _model_lock(root: Path, quiet: bool = True):
    """Hold a cross-process lock for this cache while verifying or downloading.

    The SQLite side file is persistent: removing it would let waiters lock
    different inodes. Resolve the directory so aliases share the same lock.
    """
    lock = _acquire_model_lock(root, quiet)
    try:
        yield
    finally:
        if lock is not None:
            lock.close()


def missing_files(vision: bool, root: Path | None = None) -> list[ModelFile]:
    root = root or model_dir()
    return [f for f in needed_files(vision) if not _is_verified(root / f.path, f)]


def _ssl_context() -> ssl.SSLContext:
    """Default TLS context, with certifi's CA bundle when the system has none.

    The python.org macOS installer ships without CA certificates until the
    user runs "Install Certificates.command"; certifi fills that gap.
    """
    ctx = ssl.create_default_context()
    paths = ssl.get_default_verify_paths()
    has_system_cas = any(
        x and os.path.exists(x)
        for x in (paths.cafile, paths.capath, os.environ.get("SSL_CERT_FILE"))
    )
    if not has_system_cas:
        try:
            import certifi

            ctx.load_verify_locations(cafile=certifi.where())
        except (ImportError, OSError):
            pass
    return ctx


class _Truncated(Exception):
    pass


def _download(f: ModelFile, dest: Path, quiet: bool, attempts: int = 5) -> None:
    """Download f to dest via a .part file, resuming after a cut connection."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have >= f.size:
            # Complete (or too long) but never moved into place: check it.
            if have == f.size and sha256_file(part) == f.sha256:
                os.replace(part, dest)
                return
            part.unlink()
            have = 0
        try:
            _fetch(f, part, have, quiet)
        except _Truncated as e:
            if attempt == attempts - 1:
                got = part.stat().st_size if part.exists() else 0
                why = f": {e.__cause__}" if e.__cause__ else ""
                raise urllib.error.URLError(
                    f"connection lost ({got} of {f.size} bytes{why}); "
                    "run the command again to resume"
                ) from None
            time.sleep(min(2**attempt, 10) * RETRY_DELAY)
            continue
        os.replace(part, dest)
        return


def _fetch(f: ModelFile, part: Path, have: int, quiet: bool) -> None:
    headers = {"User-Agent": f"anygrep/{__version__}"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(file_url(f), headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60, context=_ssl_context())
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # range not satisfiable: start over
            part.unlink(missing_ok=True)
            raise _Truncated from e
        raise
    with resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range; start over
        mode = "ab" if have else "wb"
        with (
            open(part, mode) as out,
            tqdm(
                total=f.size,
                initial=have,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=Path(f.path).name,
                disable=quiet,
                leave=False,
            ) as bar,
        ):
            while True:
                try:
                    b = resp.read(CHUNK)
                except (OSError, http.client.HTTPException) as e:
                    # A reset or timed-out connection (TimeoutError and
                    # ConnectionResetError are OSErrors): keep what arrived.
                    raise _Truncated from e
                if not b:
                    break
                have += len(b)
                if have > f.size:
                    # More than the file can hold (a broken mirror or proxy):
                    # stop before it fills the disk.
                    out.close()
                    part.unlink(missing_ok=True)
                    raise ModelUnavailableError(
                        f"{f.path}: the server sent more than the expected {f.size} bytes"
                    )
                out.write(b)
                bar.update(len(b))
    # urllib does not raise when the server closes the connection early.
    if part.stat().st_size < f.size:
        raise _Truncated


def ensure_model(vision: bool, quiet: bool = False) -> Path:
    """Return the model directory, downloading missing files if allowed."""
    root = model_dir()
    todo = missing_files(vision, root)
    if not todo:
        return root
    if offline():
        # Nothing can be downloaded: no lock (and no lock file) is needed.
        still = [f for f in todo if not _verify(root / f.path, f)]
        if still:
            raise ModelUnavailableError(_offline_message(still, root))
        return root
    with _model_lock(root, quiet):
        return _ensure_model_locked(vision, quiet, root)


def _ensure_model_locked(vision: bool, quiet: bool, root: Path) -> Path:
    # A different process may have completed the files while we waited.
    todo = missing_files(vision, root)
    if not todo:
        return root
    # Files may already be present (copied manually) but not yet verified.
    still: list[ModelFile] = []
    for f in todo:
        if not _verify(root / f.path, f):
            still.append(f)
    if not still:
        return root
    total_mb = sum(f.size for f in still) / 2**20
    if offline():
        raise ModelUnavailableError(_offline_message(still, root))
    if not quiet:
        print(
            f"Downloading model files ({total_mb:.0f} MB, one time) from {MODEL_REPO} to {root}",
            file=sys.stderr,
        )
    for f in still:
        dest = root / f.path
        try:
            _download(f, dest, quiet)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            hint = ""
            if "CERTIFICATE_VERIFY_FAILED" in str(e) and sys.platform == "darwin":
                hint = (
                    "\nPython has no CA certificates. With the python.org installer, run "
                    "'Install Certificates.command' from its folder in Applications.\n"
                )
            if dest.with_name(dest.name + ".part").exists() and "resume" not in str(e):
                hint += "\nThe partial download is kept; run the command again to resume.\n"
            raise ModelUnavailableError(
                f"Could not download {f.path}: {e}\n{hint}\n" + _offline_message(still, root)
            ) from e
        if not _verify(dest, f):
            dest.unlink(missing_ok=True)
            raise ModelUnavailableError(
                f"Checksum mismatch for {f.path}; the file was removed. Please retry."
            )
    return root


def _offline_message(files: Iterable[ModelFile], root: Path) -> str:
    lines = [
        "The model files are not available locally and downloading is not possible",
        "(offline mode via ANYGREP_OFFLINE/HF_HUB_OFFLINE, or no network).",
        "",
        "To use anygrep offline, download these files on a connected machine:",
    ]
    for f in files:
        lines.append(f"  {file_url(f)}")
    lines += [
        "",
        f"and place them under: {root}",
        "keeping the relative paths (e.g. onnx/model_quantized.onnx).",
        "Alternatively set ANYGREP_MODEL_DIR to a directory with that layout.",
    ]
    return "\n".join(lines)
