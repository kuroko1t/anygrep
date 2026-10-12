from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from anygrep.extract import (
    SkipFile,
    decode_text,
    html_to_text,
    kind_of,
    read_pdf_pages,
    read_text_file,
)
from anygrep.paths import true_case
from anygrep.walker import WalkStats, WalkTruncated, walk

from .conftest import write_pdf


def names(entries) -> set[str]:
    return {e.path.name for e in entries}


def test_walk_respects_hidden_ignore_and_skip_dirs(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a")
    (tmp_path / ".hidden.md").write_text("h")
    (tmp_path / ".secret").mkdir()
    (tmp_path / ".secret" / "s.md").write_text("s")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "n.md").write_text("n")
    (tmp_path / "venv").mkdir()
    (tmp_path / "venv" / "pyvenv.cfg").write_text("")
    (tmp_path / "venv" / "v.md").write_text("v")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "keep.md").write_text("k")
    (tmp_path / "sub" / "drop.md").write_text("d")
    (tmp_path / "sub" / ".gitignore").write_text("drop.md\n")
    (tmp_path / ".gitignore").write_text("build/\n*.log.md\n")
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "b.md").write_text("b")
    (tmp_path / "x.log.md").write_text("x")
    (tmp_path / "prog.exe").write_bytes(b"MZ")
    (tmp_path / "empty.md").write_text("")
    stats = WalkStats()
    got = names(walk([tmp_path], stats=stats))
    assert got == {"a.md", "keep.md"}
    assert stats.unsupported >= 1


def test_anygrepignore_and_explicit_file(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.md").write_text("b")
    (tmp_path / ".anygrepignore").write_text("b.md\n")
    assert names(walk([tmp_path])) == {"a.md"}
    # an explicitly given file is indexed even if ignored by the folder rules
    assert names(walk([tmp_path / "b.md"])) == {"b.md"}


def test_size_limit_and_extra_ext(tmp_path: Path) -> None:
    (tmp_path / "big.md").write_text("x" * 2000)
    (tmp_path / "code.py").write_text("print(1)")
    assert names(walk([tmp_path], max_size={"text": 1000})) == set()
    assert names(walk([tmp_path], frozenset({".py"}))) == {"big.md", "code.py"}


def test_kind_of() -> None:
    assert kind_of(Path("a.MD")) == "text"
    assert kind_of(Path("a.pdf")) == "pdf"
    assert kind_of(Path("a.JPEG")) == "image"
    assert kind_of(Path("a.webp")) == "image"
    assert kind_of(Path("a.zip")) is None


def test_decode_text_variants() -> None:
    assert decode_text("héllo".encode()) == "héllo"
    assert decode_text(b"\xef\xbb\xbfbom") == "bom"
    assert decode_text("caf\xe9 au lait, tr\xe8s bien".encode("cp1252")).startswith("café")
    assert decode_text("utf16".encode("utf-16")) == "utf16"
    with pytest.raises(SkipFile):
        decode_text(b"abc\x00def")


def test_html_and_newlines(tmp_path: Path) -> None:
    assert "Hello" in html_to_text("<html><script>x=1</script><p>Hello &amp; bye</p></html>")
    assert "x=1" not in html_to_text("<script>x=1</script>")
    p = tmp_path / "a.txt"
    p.write_bytes(b"one\r\ntwo\rthree")
    assert read_text_file(p) == "one\ntwo\nthree"


def test_pdf_pages(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    write_pdf(p, [["First page", "alpha"], ["Second page", "beta (gamma)"]])
    pages = list(read_pdf_pages(p))
    assert [pg.number for pg in pages] == [1, 2]
    assert "alpha" in pages[0].text
    assert "beta (gamma)" in pages[1].text


@pytest.mark.skipif(sys.platform != "linux", reason="needs bytes file names")
def test_undecodable_file_name_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "ok.md").write_text("ok")
    with open(os.fsencode(tmp_path) + b"/bad\xff.md", "w") as f:
        f.write("x")
    assert names(walk([tmp_path])) == {"ok.md"}


needs_posix_perms = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)


@needs_posix_perms
def test_unreadable_folder_does_not_stop_the_walk(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "b.md").write_text("b")
    locked.chmod(0)
    try:
        stats = WalkStats()
        assert names(walk([tmp_path], stats=stats)) == {"a.md"}
        assert locked in stats.unreadable_dirs
    finally:
        locked.chmod(0o755)


def test_nested_ignore_file_can_reinclude(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.md\n")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / ".gitignore").write_text("!keep.md\n")
    (tmp_path / "d" / "keep.md").write_text("k")
    (tmp_path / "d" / "drop.md").write_text("d")
    (tmp_path / "top.md").write_text("t")
    assert names(walk([tmp_path])) == {"keep.md"}


def test_ignore_files_above_a_subfolder_root_apply(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".gitignore").write_text("secret.md\n")
    sub = repo / "sub"
    sub.mkdir()
    (sub / "ok.md").write_text("ok")
    (sub / "secret.md").write_text("pin 1234")
    assert names(walk([repo])) == {"ok.md"}
    assert names(walk([sub])) == {"ok.md"}
    # without a git repository, an enclosing indexed root bounds the lookup
    plain = tmp_path / "plain"
    (plain / "sub").mkdir(parents=True)
    (plain / ".gitignore").write_text("secret.md\n")
    (plain / "sub" / "secret.md").write_text("x")
    (plain / "sub" / "ok.md").write_text("x")
    # This fixture is meant to have no enclosing repository. Some test hosts
    # place one above their temporary directory, so isolate only this case.
    real_exists = Path.exists

    def no_external_git(path: Path) -> bool:
        if path.name == ".git" and path.parent in plain.parents:
            return False
        return real_exists(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "exists", no_external_git)
        assert names(walk([plain / "sub"])) == {"ok.md", "secret.md"}
    assert names(walk([plain / "sub"], known_roots=frozenset({plain}))) == {"ok.md"}


def test_unsupported_explicit_file_is_reported(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("print(1)")
    stats = WalkStats()
    assert names(walk([tmp_path / "script.py"], stats=stats)) == set()
    assert stats.unsupported_roots == [tmp_path / "script.py"]


def test_user_size_limit_holds_existing_files(tmp_path: Path) -> None:
    (tmp_path / "big.md").write_text("x" * 2000)
    stats = WalkStats()
    assert names(walk([tmp_path], max_size={"text": 1000}, stats=stats)) == set()
    assert stats.held == {str(tmp_path / "big.md")}


def test_macos_hidden_flag(monkeypatch) -> None:
    import stat as stat_mod

    from anygrep import walker

    class St:
        st_flags = stat_mod.UF_HIDDEN

    monkeypatch.setattr(walker.sys, "platform", "darwin")
    assert walker._is_hidden("Library", St())  # type: ignore[arg-type]
    St.st_flags = 0
    assert not walker._is_hidden("Documents", St())  # type: ignore[arg-type]


def test_true_case_restores_disk_spelling(tmp_path: Path) -> None:
    (tmp_path / "Notes" / "Work").mkdir(parents=True)
    typed = str(tmp_path / "notes" / "WORK")
    assert true_case(typed) == str(tmp_path / "Notes" / "Work")
    assert true_case(str(tmp_path / "Notes" / "missing")) == str(tmp_path / "Notes" / "missing")


def test_named_root_excluded_by_rules_above_is_indexed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".gitignore").write_text("private/\n*.log\n")
    (repo / "private" / "sub").mkdir(parents=True)
    (repo / "private" / "a.md").write_text("a")
    (repo / "private" / "sub" / "b.md").write_text("b")
    assert names(walk([repo])) == set()
    stats = WalkStats()
    assert names(walk([repo / "private"], stats=stats)) == {"a.md", "b.md"}
    assert stats.ignored_roots == [(repo / "private", repo)]
    # A folder below an excluded one counts as excluded too.
    stats = WalkStats()
    assert names(walk([repo / "private" / "sub"], stats=stats)) == {"b.md"}
    assert stats.ignored_roots == [(repo / "private" / "sub", repo)]
    # A root that is not excluded keeps the rules from above.
    (repo / "notes").mkdir()
    (repo / "notes" / "n.md").write_text("n")
    (repo / "notes" / "debug.log").write_text("x")
    stats = WalkStats()
    assert names(walk([repo / "notes"], stats=stats)) == {"n.md"}
    assert stats.ignored_roots == []


def test_dotfiles_home_does_not_hide_named_folder(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".git").mkdir(parents=True)
    (home / ".gitignore").write_text("*\n!.bashrc\n")
    (home / "notes").mkdir()
    (home / "notes" / "n.md").write_text("hello")
    assert names(walk([home / "notes"])) == {"n.md"}


def test_html_stripping_is_linear() -> None:
    # Unclosed <script> tags and '<' without '>' used to rescan to the end of
    # the file for every '<' (quadratic: tens of seconds for a few hundred KB).
    import time

    t = time.perf_counter()
    assert html_to_text("<script>x" * 40000) == "x" * 40000
    assert html_to_text("a<b " * 100000).startswith("a<b a<b ")
    assert time.perf_counter() - t < 2.0
    assert html_to_text("1<script>\nx</SCRIPT >2<style>y</style><p\n>3") == "1\n2\n3"
    assert html_to_text("a <> b") == "a <> b"


def test_html_block_boundaries_and_quoted_attributes() -> None:
    assert html_to_text("<p>Hello</p><p>world</p>").split() == ["Hello", "world"]
    assert html_to_text("one<br>two<div>three</div>four").split() == ["one", "two", "three", "four"]
    assert html_to_text("hel<em>lo</em> world") == "hello world"
    assert html_to_text('<p title="1 > 0">Hello</p>world').split() == ["Hello", "world"]
    assert html_to_text("<p\nclass='x'>one</p>\ntwo").count("\n") == 2


def test_html_apostrophes_outside_attribute_values_are_text() -> None:
    def words(markup: str) -> list[str]:
        return html_to_text(markup).split()

    page = "<!-- it's generated -->\n<p>Refund policy</p>\n<p>Don't forget</p>\n<p>Contact</p>\n"
    assert words(page) == ["Refund", "policy", "Don't", "forget", "Contact"]
    assert words("<html>\n<!-- Don't edit -->\n<p>Vacation policy</p>\n</html>") == [
        "Vacation",
        "policy",
    ]
    assert words("<img alt=Bob's src=x.png> hello</p><p>Bob's cat</p>") == ["hello", "Bob's", "cat"]
    assert "Bob's cat" in html_to_text("<p>if a<b then it's true</p><p>Bob's cat</p><p>end</p>")
    assert "<p>" not in html_to_text("<p>if a<b then it's true</p><p>more</p><p>x</p>")
    assert words("<a title='Don't'>link</a><p>first</p>") == ["link", "first"]
    assert words("<a href=/it's-here>text</a><p>Bob's three</p>") == ["text", "Bob's", "three"]
    assert words("<!-- a > b -->kept") == ["kept"]
    assert words("<a title=\"x > y\" href='/a>b'>link</a>") == ["link"]
    assert words('<a title="unclosed>link</a>') == ["link"]


def test_walk_budget_stops_during_scandir_and_closes_it(tmp_path: Path, monkeypatch) -> None:
    from anygrep import walker

    root = tmp_path / "root"
    root.mkdir()
    consumed = []
    closed = []

    class Scan:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append(True)

        def __iter__(self):
            for i in range(100):
                consumed.append(i)
                yield object()

    monkeypatch.setattr(walker.os, "scandir", lambda _: Scan())
    with pytest.raises(WalkTruncated) as error:
        list(walk([root], max_entries=2))
    assert not error.value.by_time
    assert error.value.entries == 2
    assert consumed == [0, 1]
    assert closed == [True]


def test_walk_directory_order_is_sorted(tmp_path: Path) -> None:
    for name in ("z.md", "a.md", "m.md"):
        (tmp_path / name).write_text(name)
    assert [e.path.name for e in walk([tmp_path])] == ["a.md", "m.md", "z.md"]


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_ignore_files_that_are_not_regular_files_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "r"
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("x", encoding="utf-8")
    (root / "sub" / "b.md").write_text("x", encoding="utf-8")
    os.mkfifo(root / ".ignore")  # reading it would block forever
    (tmp_path / "rules").write_text("a.md\n", encoding="utf-8")
    os.symlink(tmp_path / "rules", root / ".gitignore")  # could point at /dev/zero
    (root / "sub" / ".anygrepignore").write_text("b.md\n", encoding="utf-8")
    assert names(walk([root])) == {"a.md"}


def test_large_ignore_file_is_read_up_to_a_limit(tmp_path: Path, monkeypatch) -> None:
    from anygrep import walker

    monkeypatch.setattr(walker, "MAX_IGNORE_FILE_BYTES", 64)
    root = tmp_path / "r"
    root.mkdir()
    for n in ("a.md", "b.md"):
        (root / n).write_text("x", encoding="utf-8")
    (root / ".gitignore").write_text("a.md\n" + "#" * 100 + "\nb.md\n", encoding="utf-8")
    assert names(walk([root])) == {"b.md"}


def test_combined_ignore_rules_decide_like_pathspec() -> None:
    import random

    import pathspec

    from anygrep.walker import _Rules

    rng = random.Random(0)
    pieces = ["*.log", "build/", "/top", "docs/**/x", "a?c", "[ab]*.md", "!keep.log", "!build/",
              "name.txt", "**/deep", "sub/*.md", "# comment", "", "\\#hash", "dir/",
              "*/", "**/", "!*/"]  # fmt: skip
    names = ["a.log", "keep.log", "build", "top", "x", "abc", "b.md", "name.txt", "deep", "#hash"]
    for _ in range(200):
        lines = rng.sample(pieces, rng.randint(1, 8))
        rules, spec = _Rules(lines), pathspec.GitIgnoreSpec.from_lines(lines)
        assert rules.fast
        for _ in range(30):
            parts = [
                rng.choice(["docs", "sub", "dir", "q", *names]) for _ in range(rng.randint(1, 4))
            ]
            rel = "/".join(parts) + rng.choice(["", "/"])
            assert rules.check(rel) == spec.check_file(rel).include, (lines, rel)


@pytest.mark.parametrize("rules", ["*/\n", "**/\n"])
def test_unanchored_folder_rules_ignore_every_subfolder(tmp_path: Path, rules: str) -> None:
    (tmp_path / ".gitignore").write_text(rules)
    (tmp_path / "top.md").write_text("t")
    (tmp_path / "private" / "sub").mkdir(parents=True)
    (tmp_path / "private" / "a.md").write_text("a")
    (tmp_path / "private" / "sub" / "b.md").write_text("b")
    assert names(walk([tmp_path])) == {"top.md"}


def test_negated_unanchored_folder_rule_reincludes(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*/\n!*/\n")
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "a.md").write_text("a")
    assert names(walk([tmp_path])) == {"a.md"}
