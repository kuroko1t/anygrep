from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from anygrep import cli, model
from anygrep.config import ModelFile
from anygrep.search import path_matcher, snippet
from anygrep.store import Store

from .conftest import jsonl


def test_cli_end_to_end(env, corpus: Path, capsys, tty) -> None:
    assert cli.main(["index", str(corpus), "-q"]) == 0
    capsys.readouterr()
    assert cli.main(["wifi password", "--everywhere", "--json", "-m", "2"]) == 0
    out = jsonl(capsys.readouterr().out)
    assert out[0]["path"].endswith("manual.pdf") and out[0]["page"] == 2
    assert out[0]["type"] == "pdf" and "wifi" in out[0]["snippet"].lower()

    tty()
    assert cli.main(["bread", "--everywhere", "--type", "text", "-m", "1"]) == 0
    text = capsys.readouterr().out
    assert "bread.md:" in text and "0." in text

    assert cli.main(["status", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["counts"]["image"]["ok"] == 2 and st["image_budget"] == 140

    # re-index with no paths updates known folders
    assert cli.main(["index", "-q"]) == 0
    assert cli.main(["remove", str(corpus / "notes")]) == 0
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    assert "text" not in json.loads(capsys.readouterr().out)["counts"]

    assert cli.main(["reset", "--yes"]) == 0
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["exists"] is False


def test_cli_search_without_index(env, capsys) -> None:
    assert cli.main(["--everywhere", "anything"]) == 2
    assert "index is empty" in capsys.readouterr().err
    assert cli.main(["--no-index", "anything"]) == 2
    assert "no index at" in capsys.readouterr().err


def test_cli_image_query(env, corpus: Path, capsys) -> None:
    cli.main(["index", str(corpus), "-q"])
    capsys.readouterr()
    assert cli.main(["--image", str(corpus / "red.png"), str(corpus), "--weak", "--json"]) == 0
    out = jsonl(capsys.readouterr().out)
    # the query image itself is excluded from results
    assert all(not h["path"].endswith("red.png") for h in out)


def test_path_matcher(tmp_path: Path) -> None:
    m = path_matcher("*.md")
    assert m("/a/b/c.md") and not m("/a/b/c.pdf")
    m = path_matcher("notes/*")
    assert m("/home/u/notes/x.md")
    m = path_matcher(str(tmp_path))
    assert m(str(tmp_path / "x.md")) and not m(str(tmp_path) + "-other/x.md")


def test_snippet() -> None:
    text = "intro " * 50 + "the router password is here " + "outro " * 50
    s = snippet(text, "router password", 80)
    assert "router" in s and s.startswith("...") and s.endswith("...") and len(s) <= 80
    for q in ("intro", "outro", "router", None):
        assert len(snippet(text, q, 80)) <= 80
    assert snippet(None, "x") == ""
    assert snippet("short", None) == "short"


def test_snippet_with_wide_characters() -> None:
    from anygrep.search import display_width

    text = "その他の内容。" * 20 + "空き家の調査を自治会の協力で進める。" + "別の話題です。" * 30
    s = snippet(text, "空き家の実態調査の進め方", 76)
    assert "空き家" in s and display_width(s) <= 76
    for q in (None, "nothing", "その他"):
        assert display_width(snippet(text, q, 60)) <= 60
    mixed = "note " * 40 + "東京の天気予報 forecast " + "日本語" * 40
    assert display_width(snippet(mixed, "天気予報", 50)) <= 50


def test_model_verify_and_offline(tmp_path: Path, monkeypatch) -> None:
    content = b"hello model"
    f = ModelFile("onnx/x.onnx", len(content), hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(model, "MODEL_FILES", (f,))
    monkeypatch.setenv("ANYGREP_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("ANYGREP_OFFLINE", "1")
    assert model.missing_files(vision=False) == [f]
    with pytest.raises(model.ModelUnavailableError) as ei:
        model.ensure_model(vision=False, quiet=True)
    assert "onnx/x.onnx" in str(ei.value)
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "x.onnx").write_bytes(content)
    assert model.ensure_model(vision=False, quiet=True) == tmp_path
    assert model.missing_files(vision=False) == []
    # a corrupted file is detected again
    bad = tmp_path / "onnx" / "x.onnx"
    bad.write_bytes(b"hello modeX")
    st = bad.stat()
    os.utime(bad, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert model.missing_files(vision=False) == [f]
    with pytest.raises(model.ModelUnavailableError):
        model.ensure_model(vision=False, quiet=True)


def test_cli_first_run_without_paths_creates_no_index(env, tmp_path: Path, capsys) -> None:
    assert cli.main(["index"]) == 2
    assert cli.main(["index", str(tmp_path / "nope")]) == 2
    assert not (tmp_path / "home").exists() or not any((tmp_path / "home").rglob("*.sqlite"))
    assert cli.main(["--everywhere", "foo"]) == 2
    assert "index is empty" in capsys.readouterr().err
    assert not (tmp_path / "home").exists() or not any((tmp_path / "home").rglob("*.sqlite"))


def test_cli_empty_index_hint(env, tmp_path: Path, capsys) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["index", str(empty), "-q"]) == 0
    capsys.readouterr()
    assert cli.main(["--everywhere", "foo"]) == 2
    assert "is empty" in capsys.readouterr().err
    assert cli.main(["foo", str(empty)]) == 2
    assert "nothing indexed under" in capsys.readouterr().err
    assert env.calls["query"] == 0  # the model was not needed


def test_cli_single_file_root_is_updated_and_pruned(env, corpus: Path, tmp_path: Path, capsys):
    lone = tmp_path / "lone.md"
    lone.write_text("a lonely note about kayaks\n")
    assert cli.main(["index", str(corpus / "notes"), str(lone), "-q"]) == 0
    assert cli.main(["status", "--json"]) == 0
    assert str(lone) in json.loads(capsys.readouterr().out)["roots"]
    lone.unlink()
    assert cli.main(["index"]) == 0
    assert "1 removed" in capsys.readouterr().err
    assert cli.main(["status", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert str(lone) not in st["roots"] and st["counts"]["text"]["ok"] == 2


def test_cli_unsupported_file_warns(env, tmp_path: Path, capsys) -> None:
    (tmp_path / "script.py").write_text("print(1)")
    assert cli.main(["index", str(tmp_path / "script.py"), "-q"]) == 0
    assert "--ext .py" in capsys.readouterr().err


def test_cli_image_query_errors_and_symlink(env, corpus: Path, tmp_path: Path, capsys) -> None:
    link = tmp_path / "lnk"
    try:
        link.symlink_to(corpus, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not available")
    assert cli.main(["index", str(link), "-q"]) == 0
    capsys.readouterr()
    assert cli.main(["--image", str(link / "red.png"), "--everywhere", "--weak", "--json"]) == 0
    out = jsonl(capsys.readouterr().out)
    assert out and all(not h["path"].endswith("red.png") for h in out)
    assert cli.main(["--image", str(corpus / "manual.pdf"), "--everywhere"]) == 2
    assert "cannot read image" in capsys.readouterr().err
    # Over Pillow's decompression-bomb limit: a message, not a traceback.
    from PIL import Image

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Image, "MAX_IMAGE_PIXELS", 1000)
        assert cli.main(["--image", str(corpus / "red.png"), "--everywhere"]) == 2
    assert "cannot read image" in capsys.readouterr().err


def test_cli_no_images_keeps_budget(env, corpus: Path, capsys) -> None:
    assert cli.main(["index", str(corpus), "-q"]) == 0
    assert cli.main(["index", str(corpus), "-q", "--image-tokens", "280", "--no-images"]) == 0
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["image_budget"] == 140 and st["image_budgets_indexed"] == {"140": 2}


def test_cli_open_without_opener(env, corpus: Path, monkeypatch, capsys) -> None:
    cli.main(["index", str(corpus), "-q"])
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["bread", "--everywhere", "--open"]) == 0
    assert "Cannot open" in capsys.readouterr().err


def test_path_matcher_anchored_globs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes").mkdir()
    target = str((tmp_path / "notes" / "a.md").resolve())
    assert path_matcher("./notes/*")(target)
    assert path_matcher("notes/*")(target)
    assert not path_matcher("./other/*")(target)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert path_matcher("~/notes/*.md")(target)
    assert not path_matcher("~/notes/*.pdf")(target)


class _Server:
    """Local HF_ENDPOINT that can cut the first response short."""

    def __init__(
        self, content: bytes, cut_first_at: int | None = None, mode: str | None = None
    ) -> None:
        """mode: "reset" resets the first connection after cut_first_at bytes,
        "cut_all" cuts every response after cut_first_at bytes, "no_range"
        ignores Range headers, "416" rejects every Range request."""
        import http.server
        import socket
        import struct
        import threading

        self.content = content
        self.requests: list[str | None] = []
        srv = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                rng = self.headers.get("Range")
                srv.requests.append(rng)
                if mode == "no_range":
                    rng = None
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                if start >= len(srv.content) or (rng and mode == "416"):
                    self.send_response(416)
                    self.end_headers()
                    return
                body = srv.content[start:]
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                first = len(srv.requests) == 1
                if cut_first_at is not None and (first or mode == "cut_all"):
                    body = body[:cut_first_at]
                self.wfile.write(body)
                self.wfile.flush()
                if mode == "reset" and first:
                    linger = struct.pack("ii", 1, 0)  # close with RST
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
                self.close_connection = True

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def model_env(tmp_path: Path, monkeypatch):
    content = bytes(range(256)) * 400
    f = ModelFile("onnx/x.onnx", len(content), hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(model, "MODEL_FILES", (f,))
    monkeypatch.setenv("ANYGREP_MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.delenv("ANYGREP_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(model, "RETRY_DELAY", 0)
    return content, tmp_path / "m" / "onnx" / "x.onnx"


def test_download_resumes_after_early_close(model_env, monkeypatch) -> None:
    content, dest = model_env
    srv = _Server(content, cut_first_at=len(content) // 3)
    monkeypatch.setenv("HF_ENDPOINT", srv.url)
    try:
        model.ensure_model(vision=False, quiet=True)
    finally:
        srv.close()
    assert dest.read_bytes() == content
    assert srv.requests[0] is None and srv.requests[1] == f"bytes={len(content) // 3}-"


def test_full_size_part_file_is_checked(model_env, monkeypatch) -> None:
    content, dest = model_env
    srv = _Server(content)
    monkeypatch.setenv("HF_ENDPOINT", srv.url)
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True)
    try:
        part.write_bytes(content)  # complete, but never renamed
        model.ensure_model(vision=False, quiet=True)
        assert srv.requests == [] and dest.read_bytes() == content
        dest.unlink()
        model._marker(dest).unlink()
        part.write_bytes(b"x" * len(content))  # full size but corrupt
        model.ensure_model(vision=False, quiet=True)
    finally:
        srv.close()
    assert srv.requests == [None] and dest.read_bytes() == content


def _status(capsys) -> dict:
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_missing_folder_is_skipped_not_deleted(env, corpus: Path, tmp_path: Path, capsys):
    assert cli.main(["index", str(corpus), "-q"]) == 0
    before = _status(capsys)["counts"]
    moved = tmp_path / "unplugged"
    corpus.rename(moved)  # like an external drive that is not mounted
    try:
        assert cli.main(["index", "-q"]) == 0
        assert "Skipping missing folders" in capsys.readouterr().err
        assert _status(capsys)["counts"] == before
    finally:
        moved.rename(corpus)


def test_cli_interrupted_budget_change_is_finished_later(env, corpus: Path, capsys) -> None:
    Image.new("RGB", (200, 150), (20, 200, 20)).save(corpus / "green.png")
    assert cli.main(["index", str(corpus), "-q"]) == 0
    env.fail_after = 1  # the second image re-embedding is interrupted
    assert cli.main(["index", "--image-tokens", "280"]) == cli.EXIT_INTERRUPTED
    err = capsys.readouterr().err
    assert "Stopped after" in err and "Done in" not in err and "Interrupted" in err
    st = _status(capsys)
    assert st["image_budget"] == 140 and st["image_budget_pending"] == 280
    assert st["image_budgets_indexed"] == {"140": 2, "280": 1}
    store = Store(cli.default_index_path(), create=False, readonly=True)
    try:
        vectors = store.load_matrix(["image"])
        assert sorted(vectors.budgets) == [140, 140, 280]
        assert len(vectors.budgets) == len(vectors.paths)
    finally:
        store.close()
    # Search embeds image queries like most indexed images.
    env.fail_after = None
    assert cli.main(["--image", str(corpus / "red.png"), "--everywhere", "--weak", "--json"]) == 0
    assert env.image_budget == 140
    calls = env.calls["image"]
    assert cli.main(["index", "-q"]) == 0  # continues the change to 280
    assert env.calls["image"] == calls + 2
    st = _status(capsys)
    assert st["image_budget"] == 280 and st["image_budget_pending"] is None
    assert st["image_budgets_indexed"] == {"280": 3}


def test_cli_named_ignored_folder_and_empty_run(env, tmp_path: Path, capsys) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / ".gitignore").write_text("private/\n")
    (repo / "private").mkdir()
    (repo / "private" / "a.md").write_text("a private note about bread")
    assert cli.main(["index", str(repo / "private")]) == 0
    err = capsys.readouterr().err
    assert "excluded by the ignore files" in err and "1 new" in err
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / ".hidden.md").write_text("x")
    assert cli.main(["index", str(tmp_path / "empty")]) == 0
    assert "No files to index: 1 ignored" in capsys.readouterr().err


def test_cli_status_model_reset_and_guards(env, corpus: Path, monkeypatch, capsys) -> None:
    (corpus / "broken.pdf").write_bytes(b"%PDF-1.4\n")
    assert cli.main(["index", str(corpus), "-q"]) == cli.EXIT_ERROR  # broken.pdf failed
    capsys.readouterr()
    assert cli.main(["status", "--problems"]) == 0
    out = capsys.readouterr().out
    assert "Indexed paths:" in out and str(corpus) in out
    assert any(ln.startswith("image ") and ln.split()[1] == "2" for ln in out.splitlines())
    assert "broken.pdf" in out
    assert cli.main(["model"]) == 0
    out = capsys.readouterr().out
    assert "total" in out and "tokenizer.json" in out
    assert cli.main(["bread", "--everywhere", "--threads", "0"]) == 2
    assert "--threads" in capsys.readouterr().err
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(""))  # not a terminal
    assert cli.main(["reset"]) == 2
    assert "refusing" in capsys.readouterr().err
    assert _status(capsys)["exists"] is True


def test_cli_index_mismatch_asks_for_reset(env, corpus: Path, capsys) -> None:
    import sqlite3

    from anygrep.store import OLDEST_SCHEMA, SCHEMA_VERSION

    assert cli.main(["index", str(corpus), "-q"]) == 0
    idx = Path(_status(capsys)["index"])
    for key, value, hint in (
        ("schema_version", str(OLDEST_SCHEMA - 1), "anygrep reset"),
        ("schema_version", str(SCHEMA_VERSION + 1), "Upgrade anygrep"),
        ("model_revision", "x", "anygrep reset"),
    ):
        db = sqlite3.connect(str(idx))
        (old,) = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        db.execute("UPDATE meta SET value=? WHERE key=?", (value, key))
        db.commit()
        assert cli.main(["bread", str(corpus)]) == 2
        err = capsys.readouterr().err
        assert hint in err
        if "newer" in err:  # resetting would destroy an index a newer version can use
            assert "reset" not in err
        db.execute("UPDATE meta SET value=? WHERE key=?", (old, key))
        db.commit()
        db.close()
    assert cli.main(["bread", str(corpus), "-m", "1"]) == 0


def _fetch_with(srv, monkeypatch) -> None:
    monkeypatch.setenv("HF_ENDPOINT", srv.url)
    try:
        model.ensure_model(vision=False, quiet=True)
    finally:
        srv.close()


def test_download_resumes_after_connection_reset(model_env, monkeypatch) -> None:
    content, dest = model_env
    srv = _Server(content, cut_first_at=len(content) // 2, mode="reset")
    _fetch_with(srv, monkeypatch)
    assert dest.read_bytes() == content and len(srv.requests) == 2  # retried in the same run


def test_download_restarts_when_range_is_ignored(model_env, monkeypatch) -> None:
    content, dest = model_env
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(content[:1000])
    srv = _Server(content, mode="no_range")
    _fetch_with(srv, monkeypatch)
    assert dest.read_bytes() == content and srv.requests == ["bytes=1000-"]


def test_download_restarts_after_416(model_env, monkeypatch) -> None:
    content, dest = model_env
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"stale" * 100)
    srv = _Server(content, mode="416")
    _fetch_with(srv, monkeypatch)
    assert dest.read_bytes() == content and srv.requests == ["bytes=500-", None]


def test_download_gives_up_with_a_resume_hint(model_env, monkeypatch) -> None:
    content, dest = model_env
    srv = _Server(content, cut_first_at=100, mode="cut_all")
    with pytest.raises(model.ModelUnavailableError, match="run the command again to resume"):
        _fetch_with(srv, monkeypatch)
    assert dest.with_name(dest.name + ".part").stat().st_size == 500


# ---------------------------------------------------------------- console and pipe I/O
def _piped_stdin(data: bytes, encoding: str = "cp932"):
    # What Python on Windows reads from a pipe: the ANSI code page, undecodable
    # bytes kept as lone surrogates.
    return io.TextIOWrapper(io.BytesIO(data), encoding=encoding, errors="surrogateescape")


def test_interactive_query_from_powershell_pipe(env, corpus: Path, monkeypatch, capsys) -> None:
    # PowerShell with $OutputEncoding = UTF8 sends a BOM; the query used to
    # reach the tokenizer as '\udcef...' and crash it.
    assert cli.main(["index", str(corpus), "-q"]) == 0
    seen: list[str] = []
    orig = env.embed_query
    monkeypatch.setattr(env, "embed_query", lambda q: seen.append(q) or orig(q))
    data = "﻿bread 発酵\r\n\r\n".encode()
    monkeypatch.setattr(cli.sys, "stdin", _piped_stdin(data))
    capsys.readouterr()
    assert cli.main(["-Q", "--everywhere", "-m", "1"]) == 0
    assert seen == ["bread 発酵"]
    assert "bread.md" in capsys.readouterr().out


def test_read_line_decoding(monkeypatch) -> None:
    monkeypatch.setattr(cli.sys, "stdin", _piped_stdin("確定申告\n".encode("cp932")))
    monkeypatch.setattr(cli.locale, "getpreferredencoding", lambda do_setlocale=True: "cp932")
    assert cli._read_line("") == "確定申告\n"
    monkeypatch.setattr(cli.sys, "stdin", _piped_stdin(b""))
    with pytest.raises(EOFError):
        cli._read_line("")
    assert cli._clean_text("a\udcffb﻿") == "a?b"


def test_json_escapes_non_ascii_unless_utf8(monkeypatch, capsys) -> None:
    for enc, escaped in (("utf-8", False), ("cp932", True), ("cp1252", True)):
        buf = io.TextIOWrapper(io.BytesIO(), encoding=enc)
        monkeypatch.setattr(cli.sys, "stdout", buf)
        text = cli._dump_json({"path": "メモ/ü.md"})
        assert json.loads(text)["path"] == "メモ/ü.md"
        assert ("\\u" in text) == escaped


def test_windows_pipe_encoding(monkeypatch) -> None:
    import ctypes
    import types

    cp = {"value": 932}
    k32 = types.SimpleNamespace(GetConsoleOutputCP=lambda: cp["value"])
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=k32), raising=False)
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)
    monkeypatch.setattr(cli.sys, "platform", "win32")
    if cli.sys.flags.utf8_mode:
        assert cli._windows_pipe_encoding() is None
        return
    assert cli._windows_pipe_encoding() == "cp932"
    cp["value"] = 65001
    assert cli._windows_pipe_encoding() == "utf-8"
    cp["value"] = 0  # no console attached
    assert cli._windows_pipe_encoding() is None
    cp["value"] = 932
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    assert cli._windows_pipe_encoding() is None
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.delenv("PYTHONIOENCODING")
    assert cli._windows_pipe_encoding() is None


def test_reset_prompt_at_eof_and_index_in_use(env, corpus: Path, monkeypatch, capsys) -> None:
    from anygrep.store import Store

    assert cli.main(["index", str(corpus), "-q"]) == 0
    # The search command closes the index again (an open handle blocks deletion on Windows).
    closed: list[bool] = []
    orig_close = Store.close
    monkeypatch.setattr(Store, "close", lambda self: closed.append(True) or orig_close(self))
    assert cli.main(["bread", "--everywhere"]) == 0
    assert closed

    class EmptyTerminal(io.StringIO):  # e.g. NUL on Windows, which reports isatty()
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(cli.sys, "stdin", EmptyTerminal(""))
    capsys.readouterr()
    assert cli.main(["reset"]) == 2
    assert "Aborted." in capsys.readouterr().out

    def busy(self, missing_ok=False):
        raise PermissionError(13, "The process cannot access the file", str(self))

    monkeypatch.setattr(Path, "unlink", busy)
    assert cli.main(["reset", "--yes"]) == 2
    assert "Close other anygrep processes" in capsys.readouterr().err


def test_download_stops_when_the_server_sends_too_much(model_env, monkeypatch) -> None:
    content, dest = model_env
    srv = _Server(content * 3)  # a broken mirror that never stops
    monkeypatch.setenv("HF_ENDPOINT", srv.url)
    try:
        with pytest.raises(model.ModelUnavailableError, match="more than the expected"):
            model.ensure_model(vision=False, quiet=True)
    finally:
        srv.close()
    assert not dest.exists() and not dest.with_name(dest.name + ".part").exists()


def test_open_runs_only_notes_pdfs_and_images(env, tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "bread.sh").write_text("echo sourdough bread starter\n", encoding="utf-8")
    assert cli.main(["index", str(root), "-q", "--ext", ".sh"]) == 0
    opened: list[list[str]] = []
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/xdg-open")
    monkeypatch.setattr(cli.subprocess, "Popen", lambda cmd, **kw: opened.append(cmd))
    capsys.readouterr()
    assert cli.main(["sourdough bread", str(root), "--weak", "--open"]) == 0
    assert opened == [] and "only opens notes, PDFs and images" in capsys.readouterr().err
    (root / "bread.md").write_text("sourdough bread starter", encoding="utf-8")
    (root / "bread.sh").unlink()
    assert cli.main(["index", str(root), "-q"]) == 0
    assert cli.main(["sourdough bread", str(root), "--weak", "--open"]) == 0
    assert opened == [["/usr/bin/xdg-open", str(root / "bread.md")]]


def test_remove_reports_unknown_paths_and_subfolders(env, corpus: Path, capsys) -> None:
    assert cli.main(["index", str(corpus), "-q"]) == 0
    capsys.readouterr()
    assert cli.main(["remove", str(corpus / "no-such-folder")]) == 2
    assert "not in the index" in capsys.readouterr().err
    # A subfolder of an indexed folder comes back on the next update: say so.
    assert cli.main(["remove", str(corpus / "notes")]) == 0
    captured = capsys.readouterr()
    assert "Removed 2 files" in captured.out and ".anygrepignore" in captured.err
    # Removing a parent also forgets the indexed folders inside it.
    assert cli.main(["index", str(corpus / "notes"), "-q"]) == 0
    assert cli.main(["remove", str(corpus)]) == 0
    capsys.readouterr()
    assert cli.main(["status", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["roots"] == []


def test_not_an_index_file_gives_a_message(env, tmp_path: Path, capsys) -> None:
    bad = tmp_path / "notes.txt"
    bad.write_text("just text, not SQLite\n" * 100, encoding="utf-8")
    assert cli.main(["--db", str(bad), "--everywhere", "bread"]) == 2
    assert "cannot read the index" in capsys.readouterr().err


def test_snippet_replaces_control_characters() -> None:
    s = snippet("a\x1b]52;c;aGk=\x07 b \x1b[2J c\x9b31m ‮evil", None)
    assert s == "a?]52;c;aGk=? b ?[2J c?31m ?evil"


@pytest.mark.skipif(os.name != "posix", reason="control characters in file names")
def test_search_output_has_no_terminal_escapes(env, tmp_path: Path, capsys, tty) -> None:
    root = tmp_path / "c"
    root.mkdir()
    text = "sourdough bread starter \x1b]0;fake title\x07 \x1b[2J\n"
    (root / "a\x1b[31m\nfake.md").write_text(text, encoding="utf-8")
    (root / "b.md").write_text(text, encoding="utf-8")
    assert cli.main(["index", str(root), "-q"]) == 0
    capsys.readouterr()
    tty()
    assert cli.main(["sourdough bread starter", str(root), "--weak"]) == 0
    out = capsys.readouterr().out
    assert "fake title" in out and "similar" in out
    assert not any(c in out for c in "\x1b\x07\x9b")
    assert len(out.splitlines()) == 3  # the newline in the name forges no line
