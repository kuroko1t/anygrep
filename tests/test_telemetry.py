"""onnxruntime usage telemetry stays off: no device id written under the home
directory and no outbound connection while the model loads and runs."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from anygrep.config import model_dir
from anygrep.model import missing_files

ROOT = Path(__file__).resolve().parents[1]

CHILD = """
import sys, time
sys.path.insert(0, {src!r})
from anygrep.config import model_dir
from anygrep.embedder import OnnxEmbedder
e = OnnxEmbedder(model_dir(), threads=2)
e.embed_query("hello")
print("ready", flush=True)
time.sleep({linger})
"""


def _child_env(home: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CACHE_HOME=str(home / ".cache"),
        LOCALAPPDATA=str(home / "AppData" / "Local"),
        ANYGREP_MODEL_DIR=str(model_dir()),
        CUDA_VISIBLE_DEVICES="",
    )
    env.update(extra)
    return env


def test_package_import_turns_telemetry_off_but_keeps_a_user_value() -> None:
    code = f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import anygrep, os; print(os.environ['ORT_DISABLE_TELEMETRY'])"
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "1"
    env["ORT_DISABLE_TELEMETRY"] = "0"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "0"


def _socket_inodes(pid: int) -> set[str]:
    found = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith("socket:["):
            found.add(target[8:-1])
    return found


def _remote_tcp(pid: int) -> list[str]:
    """Remote addresses of this process's connected TCP sockets."""
    inodes = _socket_inodes(pid)
    remotes = []
    for table in ("tcp", "tcp6"):
        try:
            lines = Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            f = line.split()
            # f[2] remote address, f[3] state (0A = listening), f[9] inode
            if f[9] in inodes and f[3] != "0A":
                remotes.append(f[2])
    return remotes


@pytest.mark.model
@pytest.mark.skipif(missing_files(vision=False), reason="model files not available locally")
def test_loading_the_model_sends_no_telemetry(tmp_path: Path) -> None:
    # With telemetry on, onnxruntime 1.31 connects about 9 s after start.
    linger = 15.0
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD.format(src=str(ROOT / "src"), linger=linger)],
        env=_child_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    remotes: set[str] = set()
    try:
        if sys.platform.startswith("linux"):
            deadline = time.monotonic() + 120 + linger
            while proc.poll() is None and time.monotonic() < deadline:
                try:
                    remotes.update(_remote_tcp(proc.pid))
                except OSError:
                    break
                time.sleep(0.1)
        out, err = proc.communicate(timeout=180)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0, err
    assert "ready" in out
    assert remotes == set()
    leftovers = [
        p.relative_to(tmp_path).as_posix()
        for p in tmp_path.rglob("*")
        if "microsoft" in p.as_posix().lower() or "onnxruntime" in p.name.lower()
    ]
    assert leftovers == []
