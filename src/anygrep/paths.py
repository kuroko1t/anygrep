"""Canonical path spelling for stored roots and files.

The same folder can be reached through a symlink, or typed with a different
letter case on case-insensitive file systems (macOS, Windows). Paths are
stored and compared in one canonical form so such spellings do not create
duplicate index entries.
"""

from __future__ import annotations

import hashlib
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

CASE_INSENSITIVE_PLATFORMS = ("darwin", "win32")


def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def canonical(p: str | os.PathLike[str]) -> Path:
    """Absolute path with symlinks resolved and, where it matters, the on-disk case."""
    s = os.path.realpath(os.path.expanduser(os.fspath(p)))
    if sys.platform in CASE_INSENSITIVE_PLATFORMS:
        s = true_case(s)
    return Path(s)


def true_case(path: str) -> str:
    """Rebuild an existing absolute path with the letter case stored on disk.

    Components that cannot be listed or are not found are kept as given.
    """
    drive, rest = os.path.splitdrive(path)
    head = drive + os.sep if rest.startswith(("/", "\\")) else drive
    parts = [x for x in rest.replace("\\", "/").split("/") if x]
    cur = head
    for i, part in enumerate(parts):
        try:
            names = os.listdir(cur or ".")
        except OSError:
            return os.path.join(cur, *parts[i:]) if cur else os.path.join(*parts[i:])
        if part not in names:
            folded = part.casefold()
            match = [n for n in names if n.casefold() == folded]
            if len(match) == 1:
                part = match[0]
        cur = os.path.join(cur, part) if cur else part
    return cur or path


# Folders whose entries are drives or mounted volumes (POSIX): /mnt/c under
# WSL, /media/<user>/<label> on Linux, /Volumes/<name> on macOS.
DRIVE_DIRS = ("/mnt", "/media", "/run/media", "/Volumes")


def _wsl_mount_root() -> str | None:
    """The [automount] root of /etc/wsl.conf when it is not the default /mnt."""
    import configparser

    try:
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.read("/etc/wsl.conf", encoding="utf-8")
        root = cp.get("automount", "root", fallback="").strip().strip("\"'")
    except (OSError, configparser.Error, UnicodeDecodeError):
        return None
    return root.rstrip("/") or None


def _unwalkable(
    path: str,
    home: str,
    mod,
    ismount: Callable[[str], bool] | None = None,
    drive_dirs: Iterable[str] = (),
) -> bool:
    """`path` and `home` are canonical; `mod` is posixpath or ntpath.
    `drive_dirs` are folders whose entries are drives (POSIX only)."""
    p = mod.normcase(mod.normpath(path))
    h = mod.normcase(mod.normpath(home))
    if mod.dirname(p) == p or p == h or p == mod.dirname(h):
        return True
    for d in drive_dirs:
        d = mod.normcase(mod.normpath(d))
        if p == d:
            return True
        if not p.startswith(d.rstrip(mod.sep) + mod.sep):
            continue
        parts = p[len(d.rstrip(mod.sep)) + 1 :].split(mod.sep)
        letter = len(parts[0]) == 1  # /mnt/c: a Windows drive under WSL
        if len(parts) == 1 and letter:
            return True
        # The Windows home folders under WSL: /mnt/c/Users and /mnt/c/Users/<name>.
        if letter and len(parts) in (2, 3) and parts[1].lower() == "users":
            return True
        if ismount is not None and ismount(p):
            return True  # a mounted volume (/media/me/USB, /Volumes/Backup)
    return False


def is_unwalkable(path: str | os.PathLike[str]) -> bool:
    """True for folders a search never walks on its own: the home folder,
    its parent, file system roots (/, C:\\, a UNC share) and drives mounted
    under /mnt, /media or /Volumes (under WSL also the Windows home folders)."""
    try:
        home = str(canonical(Path.home()))
    except (OSError, RuntimeError):  # no home folder known
        home = ""
    p = str(canonical(path))
    if sys.platform == "win32":
        return _unwalkable(p, home or os.sep, os.path)
    dirs = list(DRIVE_DIRS)
    wsl = _wsl_mount_root()
    if wsl:
        dirs.append(wsl)
    return _unwalkable(p, home or os.sep, os.path, os.path.ismount, dirs)


def is_under(p: str, base: str) -> bool:
    """True when the path `p` is `base` or inside it (both canonical)."""
    a, b = os.path.normcase(p), os.path.normcase(base)
    return a == b or a.startswith(b.rstrip("/\\") + os.sep)


def search_scope(paths: Iterable[str | os.PathLike[str]]) -> list[str]:
    """The search scope for PATHs: canonical paths, without those inside
    another one (they add nothing)."""
    scope = sorted({str(canonical(p)) for p in paths}, key=len)
    return [p for i, p in enumerate(scope) if not any(is_under(p, q) for q in scope[:i])]
