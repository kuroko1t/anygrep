"""The v0.1 command line: `anygrep QUERY [PATH ...]` (docs/cli-spec, section 9)."""

from __future__ import annotations

import io
import json
import ntpath
import os
import posixpath
import re
import sqlite3
import subprocess
import sys
from pathlib import Path, PureWindowsPath

import pytest
from PIL import Image

from anygrep import cli, indexer, thresholds
from anygrep.paths import _unwalkable, canonical
from anygrep.store import Store, acquire_writer_lock, lock_path, readonly_uri

from .conftest import jsonl

JSON_KEYS = [
    "rank", "path", "type", "line_start", "line_end", "page", "score", "weak", "snippet",
    "similar",
]  # fmt: skip
NONSENSE = "qwzx plorb fnord"


@pytest.fixture
def notes(corpus: Path) -> Path:
    """The test corpus with a tax note; the fake model matches shared words."""
    (corpus / "notes" / "tax.md").write_text(
        "# Tax return\n\nThe tax return is due in March. Keep the tax receipts.\n",
        encoding="utf-8",
    )
    return corpus


@pytest.fixture
def cutoff(monkeypatch):
    """Cutoffs for the fake model: shared words pass, unrelated words do not."""
    for pair in ("text->text", "text->pdf"):
        monkeypatch.setitem(thresholds.MIN_SCORES, pair, {0: 0.3})
    monkeypatch.setitem(thresholds.MIN_SCORES, "text->image", {140: 0.3})


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def index(capsys, *paths: Path) -> None:
    assert cli.main(["index", "-q", *map(str, paths)]) == 0
    capsys.readouterr()


def db_path() -> Path:
    return cli.default_index_path()


def counts() -> dict:
    st = Store(db_path(), create=False)
    try:
        return st.counts()
    finally:
        st.close()


class _TtyIn(io.StringIO):
    def isatty(self) -> bool:
        return True


def all_tty(monkeypatch, answer: str | None) -> None:
    """stdin, stdout and stderr are terminals; stdin holds `answer`."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys, "stdin", _TtyIn(answer or ""))


# 1 ---------------------------------------------------------------- argv shim
def test_shim_query_vs_command(env, notes: Path, cutoff, capsys) -> None:
    code, out, _ = run(capsys, "sourdough bread", str(notes))
    assert code == 0 and "bread.md" in out
    assert run(capsys, "search", "sourdough bread", str(notes))[:2] == (0, out)
    code, out, _ = run(capsys, "status")
    assert code == 0 and "Indexed paths:" in out
    # A query that starts with a command word is still a query.
    code, out, _ = run(capsys, "status of the q3 plan", str(notes))
    assert code == 1 and "Indexed paths:" not in out
    for argv in (["-e", "status", str(notes)], ["--", "status", str(notes)]):
        code, out, err = run(capsys, *argv)
        assert code == 1 and out == "" and "no strong matches" in err
    for argv in (["help"], ["-h"], ["--help"], ["-t", "pdf", "-h"]):
        assert run(capsys, *argv)[:2] == (0, cli.TOP_HELP)
    code, out, _ = run(capsys, "-V")
    assert code == 0 and out.strip() == f"anygrep {cli.__version__}"
    code, out, err = run(capsys)
    assert code == 2 and out == "" and err.startswith(cli.TOP_HELP.splitlines()[0])
    assert "anygrep model [--download]" in err
    code, out, _ = run(capsys, "rm", str(notes / "notes"))
    assert code == 0 and "Removed 3 files" in out
    code, _, err = run(capsys, "--no-ind", "bread", str(notes))  # no abbreviations
    assert code == 2 and "unrecognized arguments: --no-ind" in err
    assert run(capsys, "index", "--dry", str(notes))[0] == 2


@pytest.mark.parametrize("query", ["-foo", "-5", "--", "-40 degrees", "--json"])
def test_e_takes_a_query_that_starts_with_a_dash(query: str) -> None:
    args = cli.parse_args(["search", "-e", query, "notes", "-m", "3"])
    assert args.e == query and args.query == "notes" and args.max_count == 3
    args = cli.parse_args(["search", "--image", "-pic.png", "-l"])
    assert args.image == "-pic.png" and args.files
    # The value of another option is left alone, and so is everything after --.
    args = cli.parse_args(["search", "-g", "*.md", "-e", "-q"])
    assert args.glob == ["*.md"] and args.e == "-q" and not args.quiet
    args = cli.parse_args(["search", "--", "-e", "x"])
    assert args.e is None and [args.query, *args.paths] == ["-e", "x"]


def test_t_option_before_query_is_a_search(env, notes: Path, cutoff, capsys) -> None:
    code, out, _ = run(capsys, "-t", "pdf", "wifi password", str(notes))
    assert code == 0 and out.startswith(cli.display_path(str(canonical(notes / "manual.pdf"))))


# 2 ---------------------------------------------------------------- quoting
def test_unquoted_query_is_reported(env, capsys) -> None:
    code, out, err = run(capsys, "sourdough", "bread")
    assert code == 2 and out == ""
    assert "no such file or directory: 'bread'" in err and "did you forget to quote" in err
    assert not db_path().exists()  # nothing was opened
    code, _, err = run(capsys, "--", "status", "-m", "2")
    assert code == 2 and "did you forget" not in err and "put options before --" in err
    # With -e every positional is a PATH: no quoting hint.
    code, _, err = run(capsys, "-e", "sourdough", "bread")
    assert code == 2 and "did you forget" not in err


# 3 ---------------------------------------------------------------- -n
def test_n_is_accepted_as_line_numbers(env, notes: Path, cutoff, capsys) -> None:
    """Owner override of the spec: -n (grep's line numbers) is a no-op."""
    plain = run(capsys, "bread", str(notes))
    assert plain[0] == 0
    assert run(capsys, "bread", "-n", str(notes))[:2] == plain[:2]
    assert run(capsys, "-n", "bread", str(notes))[:2] == plain[:2]
    assert run(capsys, "bread", "--line-number", str(notes))[:2] == plain[:2]
    # A count after -n is a missing PATH; the hint points to -m.
    code, _, err = run(capsys, "bread", "-n", "5")
    assert code == 2 and "use -m 5" in err
    code, _, err = run(capsys, "bread", "-n5")
    assert code == 2 and "use -m 5" in err


# 4 ---------------------------------------------------------------- -i
def test_i_is_a_no_op(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    index(capsys, notes)
    monkeypatch.chdir(notes)
    monkeypatch.setattr(cli.sys, "stdin", None)  # a prompt would fail
    code, out, err = run(capsys, "-i", "tax")
    assert code == 0 and "tax.md" in out and "> " not in err


# 5 ---------------------------------------------------------------- -m
def test_max_count(env, notes: Path, cutoff, capsys) -> None:
    code, out, _ = run(capsys, "bread", str(notes), "--weak", "--json", "-m", "2")
    assert code == 0 and len(jsonl(out)) == 2
    code, _, err = run(capsys, "bread", str(notes), "-m", "0")
    assert code == 2 and "-m" in err


# 6 ---------------------------------------------------------------- types
def test_types(env, notes: Path, cutoff, capsys) -> None:
    index(capsys, notes)
    base = ["red picture wifi bread", str(notes), "--weak", "--json"]
    out = jsonl(run(capsys, *base, "-t", "pdf,image")[1])
    assert out and {h["type"] for h in out} == {"pdf", "image"}
    out = jsonl(run(capsys, *base, "-t", "pdf", "-t", "text")[1])
    assert {h["type"] for h in out} == {"pdf", "text"}
    out = jsonl(run(capsys, *base, "-T", "image")[1])
    assert out and "image" not in {h["type"] for h in out}
    code, _, err = run(capsys, *base, "-t", "foo")
    assert code == 2 and "unknown type foo" in err
    assert run(capsys, *base, "-t", "pdf", "-T", "pdf")[0] == 2


# 7 ---------------------------------------------------------------- scope
def _two_folders(tmp_path: Path) -> tuple[Path, Path]:
    a, b = tmp_path / "A", tmp_path / "B"
    for d, name in ((a, "a.md"), (b, "b.md")):
        d.mkdir()
        (d / name).write_text("sourdough bread with a crisp crust\n", encoding="utf-8")
    return a, b


def test_scope(env, tmp_path: Path, cutoff, capsys, monkeypatch) -> None:
    a, b = _two_folders(tmp_path)
    index(capsys, a, b)
    out = jsonl(run(capsys, "sourdough bread", str(a), "--json")[1])
    assert [Path(h["path"]).name for h in out] == ["a.md"]
    monkeypatch.chdir(a)  # PATH defaults to the current folder
    out = jsonl(run(capsys, "sourdough bread", "--json")[1])
    assert [Path(h["path"]).name for h in out] == ["a.md"]
    out = jsonl(run(capsys, "sourdough bread", "--everywhere", "--json", "--no-collapse")[1])
    assert sorted(Path(h["path"]).name for h in out) == ["a.md", "b.md"]
    code, _, err = run(capsys, "sourdough bread", "--everywhere", str(a))
    assert code == 2 and "drop PATH or drop --everywhere" in err
    # A file PATH matches only that file; several PATHs are OR'd.
    out = jsonl(run(capsys, "bread", str(a / "a.md"), str(b), "--json", "--no-collapse")[1])
    assert sorted(Path(h["path"]).name for h in out) == ["a.md", "b.md"]


@pytest.mark.skipif(os.name != "posix", reason="symlinks")
def test_scope_through_a_symlink(env, tmp_path: Path, cutoff, capsys) -> None:
    a, _ = _two_folders(tmp_path)
    index(capsys, a)
    link = tmp_path / "link-to-A"
    link.symlink_to(a, target_is_directory=True)
    out = jsonl(run(capsys, "sourdough bread", str(link), "--json")[1])
    assert [Path(h["path"]).name for h in out] == ["a.md"]


# 8 ---------------------------------------------------------------- globs
def test_globs(env, notes: Path, cutoff, capsys) -> None:
    index(capsys, notes)
    base = ["bread wifi red picture", str(notes), "--weak", "--json"]

    def names(*extra: str) -> set[str]:
        return {Path(h["path"]).name for h in jsonl(run(capsys, *base, *extra)[1])}

    assert names("-g", "*.md") == {"bread.md", "tax.md"}
    assert "manual.pdf" not in names("-g", "!*.pdf") and "red.png" in names("-g", "!*.pdf")
    assert names("-g", "*.md", "-g", "!tax*") == {"bread.md"}
    assert names("-g", "*.md", "-g", "*.txt") == {"bread.md", "tax.md", "bikes.txt"}


# 9 ---------------------------------------------------------------- auto-index
def test_auto_index_within_budget(env, notes: Path, cutoff, capsys) -> None:
    code, out, err = run(capsys, "bread", str(notes))
    assert code == 0 and "bread.md" in out
    assert f"added {cli._tilde(str(canonical(notes)))} as an indexed folder (6 files)" in err
    assert "Undo: anygrep rm" in err
    st = Store(db_path(), create=False)
    assert st.roots() == [str(canonical(notes))]
    st.close()
    # A subfolder of an indexed folder does not become a root of its own.
    (notes / "notes" / "more.md").write_text("bread and butter\n", encoding="utf-8")
    code, _, err = run(capsys, "bread", str(notes / "notes"))
    assert code == 0 and "added" not in err
    st = Store(db_path(), create=False)
    assert st.roots() == [str(canonical(notes))] and st.get_file(
        str(canonical(notes / "notes" / "more.md"))
    )
    st.close()


# 10 --------------------------------------------------------------- over budget
def test_over_budget_partly_indexed(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    index(capsys, notes)
    monkeypatch.setenv("ANYGREP_INDEX_BUDGET", "0")
    (notes / "notes" / "new.md").write_text("bread pudding with raisins\n", encoding="utf-8")
    calls = env.calls["docs"]
    code, out, err = run(capsys, "bread", str(notes), "--json", "--weak")
    assert code == 0 and env.calls["docs"] == calls  # all or nothing: nothing embedded
    assert "new.md" not in out
    lines = err.strip().splitlines()
    assert len(lines) == 1 and "has new files not indexed yet" in lines[0]
    assert "searched the indexed part" in lines[0]
    assert lines[0].endswith(f"anygrep index {cli._tilde(str(canonical(notes)))}")
    assert err.rstrip().endswith(lines[0])  # after the results, the last thing shown
    # Cheap updates still happen: a deleted file leaves the index.
    (notes / "notes" / "bikes.txt").unlink()
    out = run(capsys, "bicycle chain", str(notes), "--json", "--weak")[1]
    assert "bikes.txt" not in out and counts()["text"]["ok"] == 2
    # --index-budget overrides the environment; -y indexes whatever the estimate.
    assert run(capsys, "bread", str(notes), "--index-budget", "-1")[0] == 2
    code, out, _ = run(capsys, "bread pudding", str(notes), "-y", "-l")
    assert code == 0 and "new.md" in out


# 11 --------------------------------------------------------------- refusals
def test_over_budget_nothing_indexed_not_a_terminal(env, notes: Path, cutoff, capsys) -> None:
    code, out, err = run(capsys, "bread", str(notes), "--index-budget", "0")
    assert code == 2 and out == ""
    lines = err.strip().splitlines()
    assert len(lines) == 2 and "not indexed yet" in lines[0] and lines[0].endswith("(6 files).")
    assert "anygrep index " in lines[1] and "-y" in lines[1] and "est. " not in err
    assert run(capsys, "bread", str(notes), "--index-budget", "0", "-y")[0] == 0


# 12 --------------------------------------------------------------- terminal
@pytest.mark.parametrize("answer", ["y\n", None])
def test_terminal_indexes_everything_first(
    env, notes: Path, cutoff, capsys, monkeypatch, answer
) -> None:
    # Progress is visible: no question and no budget; the whole scope is
    # indexed in the foreground, then searched. stdin is never read.
    all_tty(monkeypatch, answer)
    code, out, err = run(capsys, "bread", str(notes), "--index-budget", "0")
    assert code == 0 and "bread.md" in out
    assert "[Y/n]" not in err and "est. " not in err and "background" not in err
    assert counts()["image"]["ok"] == 2 and counts()["text"]["ok"] == 3
    assert cli.sys.stdin.read() == (answer or "")


def test_stderr_decides_not_stdout(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    # `anygrep q | head` on a terminal: the progress bar is on stderr, so it counts.
    all_tty(monkeypatch, None)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))
    code, out, _ = run(capsys, "bread", str(notes), "--index-budget", "0", "-l")
    assert code == 0 and "bread.md" in out and counts()["image"]["ok"] == 2
    # -q shows no progress: the budget rule applies.
    (notes / "notes" / "new.md").write_text("bread rolls\n", encoding="utf-8")
    code, _, _ = run(capsys, "bread rolls", str(notes), "--index-budget", "0", "-q")
    assert code == 0 and counts()["text"]["ok"] == 3


def test_terminal_interrupt_keeps_progress_and_resumes(
    env, notes: Path, cutoff, capsys, monkeypatch
) -> None:
    all_tty(monkeypatch, None)
    monkeypatch.setattr(indexer, "COMMIT_EVERY_SEC", 0.0)
    env.fail_after = 3  # Ctrl-C after the text files
    code, _, err = run(capsys, "bread", str(notes), "--index-budget", "0")
    assert code == 130
    assert err.strip().splitlines()[-1] == (
        "Interrupted. Finished files are saved; the next search or index continues."
    )
    done = counts()
    assert done["text"]["ok"] == 3 and done.get("image", {}).get("ok", 0) == 0
    env.fail_after = None
    docs = env.calls["docs"]
    code, out, _ = run(capsys, "bread", str(notes), "--index-budget", "0")
    assert code == 0 and "bread.md" in out and counts()["image"]["ok"] == 2
    assert env.calls["docs"] - docs == 2  # only the PDF's chunks: the notes were kept


def test_terminal_scan_shows_a_count(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    all_tty(monkeypatch, None)
    monkeypatch.setattr(cli, "AUTO_INDEX_PROGRESS_DELAY", 0.0)
    monkeypatch.setattr(cli, "SCAN_STATUS_EVERY", 0.0)
    monkeypatch.setenv("COLUMNS", "1000")
    code, _, err = run(capsys, "bread", str(notes))
    where = cli._tilde(str(canonical(notes)))
    assert code == 0 and "\ranygrep: scanning " in err and f" files in {where}\r" in err


def test_scan_status_keeps_the_count_on_a_narrow_terminal(monkeypatch, capsys) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "AUTO_INDEX_PROGRESS_DELAY", 0.0)
    monkeypatch.setattr(cli, "SCAN_STATUS_EVERY", 0.0)
    monkeypatch.setenv("COLUMNS", "80")
    long = "/mnt/c/Users/someone/OneDrive/Documents/Projects/notes/" + "x" * 60
    cli._ScanStatus(long, SimpleNamespace(seen=12345))()
    row = capsys.readouterr().err.strip("\r ")
    assert row.startswith("anygrep: scanning 12,345 files in /mnt/c/") and len(row) <= 79


# 13 --------------------------------------------------------------- unwalkable
def test_home_is_not_walked(env, tmp_path: Path, notes: Path, cutoff, capsys, monkeypatch):
    home = tmp_path / "userhome"
    (home / "docs").mkdir(parents=True)
    (home / "docs" / "bread.md").write_text("sourdough bread starter\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    index(capsys, home / "docs", notes)

    def no_walk(*a, **k):
        raise AssertionError("walked")

    monkeypatch.setattr(indexer, "plan", no_walk)
    monkeypatch.chdir(home)
    code, out, err = run(capsys, "sourdough bread", "-l")
    assert code == 0 and out.splitlines() == [os.path.join("docs", "bread.md")]
    assert "~ is not checked for new files (home folders and drive roots" in err
    assert "searched the indexed part" in err
    # One line, after the results, naming the indexed folder in it.
    assert err.strip().endswith("Update with: anygrep index " + os.path.join("~", "docs"))
    assert run(capsys, "sourdough bread", str(home.parent))[0] == 0  # the parent of home
    # Nothing indexed under it: exit 2, and no index file is created for it.
    other = tmp_path / "other.sqlite"
    code, _, err = run(capsys, "sourdough bread", "--db", str(other))
    lines = err.strip().splitlines()
    assert code == 2 and len(lines) == 2 and not other.exists()
    assert "nothing is indexed under ~" in lines[0] and "too large" in lines[0]
    assert f"anygrep index --db {cli._tilde(str(other))} <folder>" in lines[1]
    assert "--everywhere" not in err and err.count("too large") == 1
    # An index that has files elsewhere: the same explanation, said once.
    monkeypatch.setenv("HOME", str(tmp_path / "home2"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home2"))
    (tmp_path / "home2").mkdir()
    monkeypatch.chdir(tmp_path / "home2")
    code, _, err = run(capsys, "sourdough bread")
    assert code == 2 and err.count("too large to index automatically") == 1
    assert len(err.strip().splitlines()) == 2


def test_nothing_indexed_here_hints(env, notes: Path, tmp_path: Path, cutoff, capsys) -> None:
    empty = tmp_path / "emptydir"
    empty.mkdir()
    code, _, err = run(capsys, "bread", str(empty))  # empty index: no --everywhere hint
    assert code == 2 and "nothing indexed under" in err and "--everywhere" not in err
    index(capsys, notes)
    code, _, err = run(capsys, "bread", str(empty))
    assert code == 2 and "or search with --everywhere" in err


def test_unwalkable_paths_posix_and_windows() -> None:
    home = "/home/me"
    for p, want in (("/", True), (home, True), ("/home", True), ("/home/me/notes", False)):
        assert _unwalkable(p, home, posixpath) is want, p
    home = r"C:\Users\Me"
    for p, want in (
        ("C:\\", True),
        ("D:\\", True),
        (r"c:\users\me", True),  # case-insensitive
        (r"C:\Users", True),
        (r"C:\Users\Me\notes", False),
        ("\\\\server\\share\\", True),
        (r"\\server\share\notes", False),
    ):
        assert _unwalkable(p, home, ntpath) is want, p


def test_unwalkable_drive_mounts_posix() -> None:
    home = "/home/me"
    mounts = {"/media/me/USB", "/Volumes/Backup", "/data"}
    dirs = ("/mnt", "/media", "/run/media", "/Volumes", "/win")

    def check(p: str) -> bool:
        return _unwalkable(p, home, posixpath, mounts.__contains__, dirs)

    for p, want in (
        ("/mnt", True),
        ("/mnt/c", True),
        ("/mnt/d", True),
        ("/mnt/c/Users", True),
        ("/mnt/c/users/someone", True),  # the Windows home under WSL
        ("/mnt/c/Users/someone/notes", False),
        ("/mnt/c/projects", False),
        ("/win/c", True),  # a custom WSL automount root
        ("/media/me/USB", True),
        ("/media/me/USB/photos", False),
        ("/Volumes/Backup", True),
        ("/data", False),  # a mount point outside the drive folders
        ("/home/me/notes", False),
    ):
        assert check(p) is want, p
    # Windows paths are not affected by the POSIX drive folders.
    assert _unwalkable(r"D:\mnt\c", r"C:\Users\Me", ntpath) is False


# 14 --------------------------------------------------------------- walk cap
def test_walk_cap(env, tmp_path: Path, cutoff, capsys, monkeypatch) -> None:
    big = tmp_path / "big"
    big.mkdir()
    for i in range(20):
        (big / f"n{i}.md").write_text(f"note {i} about bread\n", encoding="utf-8")
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_ENTRIES", 5)
    where = cli._tilde(str(canonical(big)))
    code, _, err = run(capsys, "bread", str(big))
    assert code == 2 and env.calls["docs"] == 0
    lines = err.strip().splitlines()
    assert len(lines) == 2 and "not indexed yet and is too large" in lines[0]
    assert lines[1] == f"  index it once with: anygrep index {where}"
    assert "scanning" not in err and "entries" not in err
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_SECONDS", 0.0)
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_ENTRIES", 10**6)
    code, _, err = run(capsys, "bread", str(big))
    assert code == 2 and len(err.strip().splitlines()) == 2 and " s)" not in err
    # Partly indexed: the indexed part is searched, with one note after the results.
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_SECONDS", 10.0)
    index(capsys, big)
    (big / "new.md").write_text("bread rolls\n", encoding="utf-8")
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_SECONDS", 0.0)
    code, out, err = run(capsys, "bread", str(big), "-l")
    assert code == 0 and "new.md" not in out
    lines = err.strip().splitlines()
    assert len(lines) == 1 and "too large to check for new files" in lines[0]
    assert lines[0].endswith(f"Update with: anygrep index {where}")


def test_walk_cap_not_on_a_terminal(env, tmp_path: Path, cutoff, capsys, monkeypatch) -> None:
    # A terminal shows the scan and the progress: no caps, the folder is indexed.
    big = tmp_path / "big"
    big.mkdir()
    for i in range(3):
        (big / f"n{i}.md").write_text(f"note {i} about bread\n", encoding="utf-8")
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_SECONDS", 0.0)
    monkeypatch.setattr(cli, "AUTO_INDEX_MAX_ENTRIES", 1)
    all_tty(monkeypatch, None)
    code, out, err = run(capsys, "bread", str(big), "--index-budget", "0")
    assert code == 0 and "n0.md" in out and "too large" not in err


# 15 --------------------------------------------------------------- --no-index
@pytest.mark.skipif(os.name != "posix", reason="file modes")
def test_no_index_opens_read_only(env, notes: Path, cutoff, capsys) -> None:
    index(capsys, notes)
    (notes / "notes" / "new.md").write_text("bread pudding\n", encoding="utf-8")
    db = db_path()
    db.chmod(0o444)
    try:
        code, out, _ = run(capsys, "bread", str(notes), "--no-index", "--weak", "-l")
        assert code == 0 and "bread.md" in out and "new.md" not in out
    finally:
        db.chmod(0o600)
    assert counts()["text"]["ok"] == 3
    # The -wal/-shm files the read-only run created read-only do not block writes later.
    code, out, _ = run(capsys, "bread pudding", str(notes), "-l")
    assert code == 0 and "new.md" in out


def test_no_index_without_a_database(env, tmp_path: Path, notes: Path, capsys) -> None:
    missing = tmp_path / "missing.sqlite"
    code, _, err = run(capsys, "bread", str(notes), "--no-index", "--db", str(missing))
    assert code == 2 and "no index at" in err and not missing.exists()


def test_readonly_old_index_does_not_migrate(env, tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite"
    Store(db).close()
    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE files DROP COLUMN sig")
    old = Store(db, create=False, readonly=True)
    try:
        assert old._sig_col == "NULL"
        assert old.load_matrix().budgets == []
    finally:
        old.close()
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    assert "sig" not in cols


def test_readonly_uri_for_windows_paths() -> None:
    uri = readonly_uri(r"C:\Users\Me\anygrep data\index #1.sqlite", pure=PureWindowsPath)
    assert uri == "file:///C:/Users/Me/anygrep%20data/index%20%231.sqlite?mode=ro"
    uri = readonly_uri(r"\\server\share\index.sqlite", pure=PureWindowsPath)
    assert uri == "file:////server/share/index.sqlite?mode=ro"


# 16 --------------------------------------------------------------- lock busy
def test_lock_busy_searches_the_snapshot(env, notes: Path, cutoff, capsys) -> None:
    index(capsys, notes)
    (notes / "notes" / "new.md").write_text("bread pudding\n", encoding="utf-8")
    other = sqlite3.connect(str(db_path()), isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    try:
        code, out, err = run(capsys, "bread", str(notes), "-l")
        assert code == 0 and "bread.md" in out and "new.md" not in out
        assert "being updated by another anygrep process" in err
    finally:
        other.execute("ROLLBACK")
        other.close()


def test_running_index_blocks_other_writers(env, notes: Path, cutoff, capsys) -> None:
    """An `anygrep index` holds the writer lock for its whole run, not only
    while it commits: searches skip indexing, other writers exit 2."""
    index(capsys, notes)
    (notes / "notes" / "new.md").write_text("bread pudding\n", encoding="utf-8")
    lock = acquire_writer_lock(db_path())
    assert lock is not None and acquire_writer_lock(db_path()) is None
    try:
        code, out, err = run(capsys, "bread", str(notes), "-l")
        assert code == 0 and "new.md" not in out and "being updated" in err
        for argv in (["index", str(notes)], ["remove", str(notes)], ["reset", "-y"]):
            code, _, err = run(capsys, *argv)
            assert code == 2 and "being updated by another anygrep process" in err, argv
    finally:
        lock.close()
    assert counts()["text"]["ok"] == 3 and db_path().exists()
    code, out, _ = run(capsys, "bread pudding", str(notes), "-l")
    assert code == 0 and "new.md" in out
    assert run(capsys, "reset", "-y")[0] == 0
    assert not db_path().exists() and lock_path(db_path()).exists()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privileges on Windows")
def test_db_symlink_alias_uses_same_writer_lock(env, tmp_path: Path, notes: Path, capsys) -> None:
    db = tmp_path / "real.sqlite"
    alias = tmp_path / "alias.sqlite"
    assert run(capsys, "index", str(notes), "--db", str(db), "-q")[0] == 0
    alias.symlink_to(db)
    assert lock_path(alias) == lock_path(db)
    held = acquire_writer_lock(alias)
    assert held is not None
    try:
        assert acquire_writer_lock(db) is None
        for command in ("index", "reset"):
            argv = [command, "--db", str(db), "-y"]
            if command == "index":
                argv = [command, str(notes), "--db", str(db)]
            code, _, err = run(capsys, *argv)
            assert code == 2 and "being updated" in err
    finally:
        held.close()
    assert run(capsys, "reset", "--db", str(alias), "-y")[0] == 0
    assert not db.exists() and lock_path(db).exists()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "1e308", "xyz"])
def test_invalid_max_size_rejected_before_opening_index(
    env,
    tmp_path: Path,
    notes: Path,
    capsys,
    value: str,
) -> None:
    db = tmp_path / "max-size.sqlite"
    code, _, err = run(capsys, "index", str(notes), "--db", str(db), "--max-size", value)
    assert code == 2 and "--max-size" in err
    assert not db.exists() and not lock_path(db).exists()


def test_valid_max_size_indexes_files(env, tmp_path: Path, notes: Path, capsys) -> None:
    db = tmp_path / "valid-size.sqlite"
    code, _, err = run(capsys, "index", str(notes), "--db", str(db), "--max-size", "0.5", "-q")
    assert code == 0, err
    assert db.exists()
    store = Store(db, create=False, readonly=True)
    try:
        assert store.counts()["text"]["ok"] > 0
    finally:
        store.close()


def test_index_double_dash_preserves_numeric_looking_file_names(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("--max-size", "-inf"):
        Path(name).write_text("hello", encoding="utf-8")
    args = cli.parse_args(["index", "--", "--max-size", "-inf"])
    assert args.paths == ["--max-size", "-inf"] and args.max_size is None


# 17 --------------------------------------------------------------- exit codes
def test_exit_codes(env, tmp_path: Path, notes: Path, cutoff, capsys, monkeypatch) -> None:
    code, out, err = run(capsys, NONSENSE, str(notes))
    assert code == 1 and out == "" and "use --weak to see the closest" in err
    code, out, _ = run(capsys, NONSENSE, str(notes), "--weak")
    assert code == 0 and out
    empty = tmp_path / "empty.sqlite"
    Store(empty).close()
    code, _, err = run(capsys, NONSENSE, "--everywhere", "--db", str(empty))
    assert code == 2 and "index is empty" in err
    code, _, err = run(capsys, "--image", str(notes / "manual.pdf"), str(notes))
    assert code == 2 and "cannot read image" in err
    assert run(capsys, "--image", str(notes / "nope.png"), str(notes))[0] == 2
    assert run(capsys, "index", str(tmp_path / "nope"))[0] == 2
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO("n\n"))
    assert run(capsys, "reset")[0] == 2
    monkeypatch.setenv("ANYGREP_INDEX_BUDGET", "soon")
    code, _, err = run(capsys, "bread", str(notes))
    assert code == 2 and "ANYGREP_INDEX_BUDGET" in err
    monkeypatch.delenv("ANYGREP_INDEX_BUDGET")
    assert run(capsys, "bread", str(notes), "-j", "0")[0] == 2


def test_interrupt_while_auto_indexing(env, notes: Path, cutoff, capsys) -> None:
    env.fail_after = 0
    code, _, err = run(capsys, "bread", str(notes))
    assert code == 130 and "Interrupted." in err
    # The folder is kept (the next index resumes it) and the user is told so.
    where = cli._tilde(str(canonical(notes)))
    assert f"anygrep: added {where} as an indexed folder" in err
    assert f"Undo: anygrep rm {where}" in err
    assert err.index("added") < err.index("Interrupted.")


def test_interrupt_exits_130(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    index(capsys, notes)

    def stop(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(env, "embed_query", stop)
    code, _, err = run(capsys, "bread", str(notes))
    assert code == 130 and "Interrupted." in err


# 18 --------------------------------------------------------------- pipe format
def test_pipe_format(env, notes: Path, cutoff, capsys) -> None:
    code, out, _ = run(capsys, "wifi password bread red picture", str(notes), "--weak")
    lines = out.splitlines()
    assert code == 0 and len(lines) >= 4
    assert all(re.fullmatch(r"[^\n]+:\d+:.*", ln) and ":0:" not in ln for ln in lines)
    assert any(":1:[p." in ln and "manual.pdf" in ln for ln in lines)
    assert any(ln.endswith("red.png:1:[image]") for ln in lines)
    bread = cli.display_path(str(canonical(notes / "notes" / "bread.md")))
    assert any(ln.startswith(bread + ":1:") for ln in lines)
    assert "\x1b" not in out and all(len(ln.split(":", 2)[2]) <= 200 for ln in lines)
    out = run(capsys, "wifi password bread", str(notes), "--weak", "--color", "always")[1]
    assert "\x1b[" in out
    out = run(capsys, "bread", str(notes), "--color", "never")[1]
    assert "\x1b" not in out and "similar" not in out


# 19 --------------------------------------------------------------- tty format
def test_tty_format(env, notes: Path, cutoff, capsys, tty) -> None:
    index(capsys, notes)
    tty()
    code, out, _ = run(capsys, "bread", str(notes), "-t", "text")
    assert code == 0
    first = out.splitlines()[0]
    assert re.search(r"bread\.md:1$", first) and not re.search(r":\d+-\d+", out)
    code, out, err = run(capsys, NONSENSE, str(notes))
    assert code == 1 and out.startswith(cli.NO_STRONG) and "(weak)" in out
    code, out, err = run(capsys, NONSENSE, str(notes), "-t", "image", "-g", "*.nothing")
    assert code == 1 and out == "" and "No results." in err


# 20 --------------------------------------------------------------- -l / -0
def test_files_with_matches(env, notes: Path, cutoff, capsys) -> None:
    index(capsys, notes)
    strong = [h["path"] for h in jsonl(run(capsys, "bread wifi", str(notes), "--json")[1])]
    code, out, _ = run(capsys, "bread wifi", str(notes), "-l")
    assert code == 0 and out.splitlines() == [cli.display_path(p) for p in strong]
    out = run(capsys, "bread wifi", str(notes), "-l", "--chunks")[1].splitlines()
    assert len(out) == len(set(out)) and "manual.pdf" in "".join(out)
    out = run(capsys, "bread wifi", str(notes), "-l", "-0")[1]
    assert out.endswith("\0") and out.split("\0")[:-1] == [cli.display_path(p) for p in strong]
    code, _, err = run(capsys, "bread", str(notes), "-0")
    assert code == 2 and "-0 needs -l" in err
    assert run(capsys, NONSENSE, str(notes), "-l")[:2] == (1, "")


# 21 --------------------------------------------------------------- JSON Lines
def test_json_lines(env, notes: Path, cutoff, capsys) -> None:
    Image.open(notes / "red.png").save(notes / "red-copy.png")
    code, out, _ = run(capsys, "bread red picture", str(notes), "--json", "--weak")
    rows = [json.loads(line) for line in out.splitlines()]
    assert code == 0 and rows and all(list(r) == JSON_KEYS for r in rows)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    assert all(isinstance(r["similar"], list) for r in rows)
    red = next(r for r in rows if r["path"].endswith(("red.png", "red-copy.png")))
    assert len(red["similar"]) == 1 and list(red["similar"][0]) == JSON_KEYS[1:-1]
    assert all(os.path.isabs(r["path"]) for r in rows) and "\n  " not in out
    assert run(capsys, NONSENSE, str(notes), "--json")[:2] == (1, "")
    code, _, err = run(capsys, "bread", str(notes), "--json", "-l")
    assert code == 2 and "cannot be combined" in err


# 22 --------------------------------------------------------------- -q
def test_quiet(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.sys.stderr, "isatty", lambda: True, raising=False)
    assert run(capsys, "bread", str(notes), "-q") == (0, "", "")  # no progress, no notes
    assert run(capsys, NONSENSE, str(notes), "-q") == (1, "", "")
    assert run(capsys, "bread", str(notes), "-q", "--json", "-l")[0] == 2  # usage still checked
    code, out, err = run(capsys, "bread", str(notes), "-q", "-l")
    assert (code, out, err) == (0, "", "")


# 23 --------------------------------------------------------------- -Q
def test_interactive(env, notes: Path, cutoff, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"tax\n\n")))
    code, out, err = run(capsys, "-Q", str(notes))
    assert code == 0 and "tax.md" in out and re.search(r"^ 1  0\.\d{3}  ", out, re.M)
    assert "> " in err and "> " not in out
    for flag in ("--json", "-l", "-q", "--open"):
        code, _, err = run(capsys, "-Q", str(notes), flag)
        assert code == 2 and "not available with -Q" in err
    assert run(capsys, "-Q", "-e", "tax", str(notes))[0] == 2


# 24 --------------------------------------------------------------- --dry-run
def test_index_dry_run(env, notes: Path, capsys) -> None:
    code, out, _ = run(capsys, "index", "--dry-run", str(notes))
    want = f"{cli._tilde(str(canonical(notes)))}: 6 new or changed files (3 text, 1 PDF, 2 images)"
    assert code == 0 and out == want + ", 0 to remove\n"
    assert not db_path().exists()
    index(capsys, notes)
    before = counts()
    (notes / "notes" / "new.md").write_text("bread pudding\n", encoding="utf-8")
    (notes / "notes" / "bikes.txt").unlink()
    code, out, _ = run(capsys, "index", "--dry-run")
    assert code == 0 and "1 new or changed files (1 text, 0 PDF, 0 images), 1 to remove" in out
    assert counts() == before


# 25 --------------------------------------------------------------- calibration
def test_calibration(env, tmp_path: Path, notes: Path, fake_embedder, monkeypatch) -> None:
    st = Store(tmp_path / "cal.sqlite")
    assert indexer.throughput_factor(st) == indexer.CALIB_DEFAULT
    assert indexer.update_throughput(st, 10.0, 5.0) is None  # too short to tell
    assert indexer.update_throughput(st, 10.0, 30.0) == 3.0
    assert indexer.update_throughput(st, 20.0, 20.0) == 2.0  # moving average
    assert indexer.update_throughput(st, 10.0, 1000.0) == 3.0  # clamped sample: 4
    assert indexer.update_throughput(st, 100.0, 20.0) == 1.75  # 0.5 * 0.5 + 0.5 * 3
    monkeypatch.setattr(indexer, "CALIB_MIN_SECONDS", 0.0)
    st.delete_meta(indexer.CALIB_KEY)
    indexer.run_index(st, [notes], indexer.IndexOptions(quiet=True), lambda v: fake_embedder)
    lo, hi = indexer.CALIB_RANGE
    assert lo <= float(st.get_meta(indexer.CALIB_KEY)) <= hi
    st.close()


def test_status_shows_measured_speed(env, notes: Path, capsys) -> None:
    index(capsys, notes)
    st = Store(db_path())
    st.set_meta(indexer.CALIB_KEY, "1.3")
    st.close()
    assert "measured speed: 1.3x the default estimate" in run(capsys, "status")[1]
    assert json.loads(run(capsys, "status", "--json")[1])["throughput_factor"] == 1.3


# 26 --------------------------------------------------------------- stdin PATH
def test_stdin_path(env, notes: Path, capsys) -> None:
    code, _, err = run(capsys, "bread", "-")
    assert code == 2 and "reading from stdin is not supported" in err


# -------------------------------------------------------------------- others
def test_everywhere_and_no_index_do_not_write(env, notes: Path, cutoff, capsys) -> None:
    index(capsys, notes)
    (notes / "notes" / "new.md").write_text("bread pudding\n", encoding="utf-8")
    for flag in ("--everywhere", "--no-index"):
        args = ["sourdough bread", "-l", flag] + ([] if flag == "--everywhere" else [str(notes)])
        code, out, _ = run(capsys, *args)
        assert code == 0 and "new.md" not in out
    assert counts()["text"]["ok"] == 3


def test_open_only_for_results_that_passed(env, notes: Path, cutoff, capsys, monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(cli, "_open_file", opened.append)
    assert run(capsys, NONSENSE, str(notes), "--open")[0] == 1
    assert opened == []
    assert run(capsys, "bread", str(notes), "--open", "-q")[0] == 0
    assert opened == []
    assert run(capsys, "bread", str(notes), "--open")[0] == 0
    assert opened and opened[0].endswith("bread.md")


def test_windows_terminal_colour(monkeypatch) -> None:
    class Term:
        def isatty(self) -> bool:
            return True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("WT_SESSION", raising=False)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli._use_color(Term()) is False  # the classic console shows escapes as text
    assert cli._use_color(Term(), "always") is True
    monkeypatch.setenv("WT_SESSION", "1")
    assert cli._use_color(Term()) is True
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._use_color(Term()) is False and cli._use_color(Term(), "always") is True
    assert cli._use_color(Term(), "never") is False


def test_windows_paths_in_messages(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli._tilde("/data/my notes") == '"/data/my notes"'
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._tilde("/data/my notes") == "'/data/my notes'"
    assert cli._tilde(str(Path.home() / "my notes")) == f"~{os.sep}'my notes'"

    def other_drive(p, start=None):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")

    monkeypatch.setattr(cli.os.path, "relpath", other_drive)
    assert cli.display_path("D:\\notes\\a.md") == "D:\\notes\\a.md"


def _output_child(lines: int) -> list[str]:
    """A child running main() with a command that prints `lines` lines once
    its stdin is closed (so the test can close the reading end first)."""
    code = (
        "import sys\n"
        "from anygrep import cli\n"
        "def big(args):\n"
        "    sys.stdin.read()\n"
        f"    for i in range({lines}):\n"
        "        print('x' * 40)\n"
        "    return 0\n"
        "cli.cmd_model = big\n"
        "sys.exit(cli.main(['model']))\n"
    )
    return [sys.executable, "-c", code]


@pytest.mark.skipif(sys.platform == "win32", reason="a closed pipe is EINVAL there; POSIX only")
@pytest.mark.parametrize("lines", [10, 2000])  # within stdout's buffer, and past the pipe's
def test_closed_pipe_exits_0_quietly(lines: int) -> None:
    p = subprocess.Popen(
        _output_child(lines), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert p.stdin is not None and p.stdout is not None and p.stderr is not None
    p.stdout.close()  # like `| head -1` quitting early
    p.stdin.close()
    err = p.stderr.read()
    assert p.wait(timeout=60) == 0, err
    assert err == b""


@pytest.mark.skipif(not os.path.exists("/dev/full"), reason="needs /dev/full")
def test_full_disk_on_stdout_is_an_error() -> None:
    with open("/dev/full", "w") as full:
        p = subprocess.run(
            _output_child(10), stdin=subprocess.DEVNULL, stdout=full, stderr=subprocess.PIPE,
            timeout=60,
        )  # fmt: skip
    assert p.returncode == 2
    assert p.stderr.decode().strip() == "anygrep: write error: No space left on device"


@pytest.mark.parametrize("value", ["0", "-3", "abc", "1.5"])
def test_bad_anygrep_threads_is_an_error(env, capsys, monkeypatch, value: str) -> None:
    monkeypatch.setenv("ANYGREP_THREADS", value)
    code, out, err = run(capsys, "status")
    assert code == 2 and out == ""
    assert err.strip() == f"anygrep: ANYGREP_THREADS={value!r} is not a number of threads >= 1"
    # -j wins over the environment.
    code, out, _ = run(capsys, "status", "-j", "3")
    assert code == 0 and "Threads: 3" in out


def test_anygrep_threads(env, capsys, monkeypatch) -> None:
    monkeypatch.setenv("ANYGREP_THREADS", "5")
    assert "Threads: 5" in run(capsys, "status")[1]


def test_files_are_coloured_only_with_always(env, notes: Path, cutoff, capsys, monkeypatch):
    index(capsys, notes)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    out = run(capsys, "bread wifi", str(notes), "-l")[1]
    assert out and "\033[" not in out
    out = run(capsys, "bread wifi", str(notes), "-l", "--color", "always")[1]
    assert "\033[35m" in out


@pytest.mark.parametrize("argv", [[""], ["   "], ["-e", ""], ["﻿"]])
def test_empty_query_is_a_usage_error(env, notes: Path, capsys, argv: list[str]) -> None:
    code, out, err = run(capsys, *argv, str(notes))
    assert code == 2 and out == ""
    assert "the query is empty" in err
    assert not db_path().exists()  # nothing was opened or loaded


def test_hints_name_the_db_option(env, notes: Path, tmp_path: Path, cutoff, capsys) -> None:
    db = tmp_path / "other" / "index.sqlite"
    db.parent.mkdir()
    shown = cli._tilde(str(db))
    code, _, err = run(capsys, "bread", str(notes), "--db", str(db))
    assert code == 0 and f"Undo: anygrep rm --db {shown} " in err
    new = tmp_path / "new"
    new.mkdir()
    (new / "a.md").write_text("bread\n", encoding="utf-8")
    code, _, err = run(capsys, "bread", str(new), "--db", str(db), "--index-budget", "0")
    assert code == 2 and f"with: anygrep index --db {shown} " in err
    # Without --db (here $ANYGREP_DB) the hints stay short.
    code, _, err = run(capsys, "bread", str(notes))
    assert code == 0 and "Undo: anygrep rm " in err and "--db" not in err


def test_global_options_before_the_command() -> None:
    f = cli._command_first
    assert f(["-j", "2", "status"]) == ["status", "-j", "2"]
    assert f(["--db", "X", "index", "P"]) == ["index", "--db", "X", "P"]
    assert f(["-j", "2", "index", "--background", "P"]) == [
        "index", "-j", "2", "--background", "P",
    ]  # fmt: skip
    assert f(["-j2", "--db=X", "rm", "P"]) == ["rm", "-j2", "--db=X", "P"]
    assert f(["--threads=4", "status"]) == ["status", "--threads=4"]
    # A query is still a query, also when it follows options.
    assert f(["-j", "2", "tax return"]) == ["-j", "2", "tax return"]
    assert f(["-l", "status"]) == ["-l", "status"]
    assert f(["status"]) == ["status"]


def test_global_option_before_status_runs_status(env, tmp_path: Path, notes, capsys) -> None:
    index(capsys, notes)
    code, out, _ = run(capsys, "-j", "2", "status")
    assert code == 0 and "Index:" in out and "Threads: 2" in out


def test_index_exits_2_when_a_file_failed(env, corpus: Path, capsys) -> None:
    (corpus / "broken.png").write_bytes(b"\x89PNG not really")
    code, out, err = run(capsys, "index", "-q", str(corpus))
    assert code == 2 and out == "" and err == ""
    code, _, err = run(capsys, "index", str(corpus))  # not retried: nothing failed now
    assert code == 0 and "failed" not in err
    (corpus / "notes" / "more.md").write_text("# More\n\nAnother note.\n", encoding="utf-8")
    (corpus / "broken2.png").write_bytes(b"\x89PNG not really")
    code, _, err = run(capsys, "index", str(corpus))
    assert code == 2 and "1 new" in err and "1 failed" in err  # the summary still prints
