"""Skipped Office documents: one line on stderr when new files are scanned."""

from __future__ import annotations

from pathlib import Path

import pytest

import anygrep
from anygrep import cli, embedder
from anygrep.walker import WalkStats, walk

LINE = "Office documents"


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def add_documents(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("a.docx", "b.DOCX", "c.xlsx", "d.pptx", "e.odt"):
        (folder / name).write_bytes(b"PK\x03\x04 not a real document")
    (folder / "~$a.docx").write_bytes(b"lock")  # Office's lock file: not counted
    for name in ("f.zip", "g.mp4", "h.exe"):  # other unsupported files: not counted
        (folder / name).write_bytes(b"\x00\x01")


def test_walk_counts_documents_by_extension(corpus: Path) -> None:
    add_documents(corpus / "office")
    stats = WalkStats()
    list(walk([corpus], stats=stats))
    assert stats.documents == {".docx": 2, ".xlsx": 1, ".pptx": 1, ".odt": 1}
    # A document named explicitly is reported on its own, not counted here.
    stats = WalkStats()
    list(walk([corpus / "office" / "a.docx"], stats=stats))
    assert stats.documents == {} and stats.unsupported_roots


def test_documents_note_wording() -> None:
    stats = WalkStats()
    assert cli._documents_note(stats) is None
    stats.documents = {".docx": 1}
    assert cli._documents_note(stats) == (
        "anygrep: skipped 1 Office document (.docx); this format is not supported yet"
    )
    stats.documents = {".xlsx": 3, ".docx": 9, ".epub": 1, ".pptx": 3}
    assert cli._documents_note(stats) == (
        "anygrep: skipped 16 Office documents (.docx, .pptx, .xlsx, ...); "
        "these formats are not supported yet"
    )


def test_index_says_once_per_run_with_new_files(env, corpus: Path, capsys) -> None:
    add_documents(corpus / "office")
    want = (
        "anygrep: skipped 5 Office documents (.docx, .odt, .pptx, ...); "
        "these formats are not supported yet"
    )
    code, out, err = run(capsys, "index", str(corpus))
    assert code == 0 and out == "" and want in err.splitlines()
    # Nothing new: no line.
    code, out, err = run(capsys, "index", str(corpus))
    assert code == 0 and LINE not in err
    (corpus / "notes" / "new.md").write_text("A new note.\n", encoding="utf-8")
    code, _, err = run(capsys, "index", str(corpus))
    assert code == 0 and want in err.splitlines()
    (corpus / "notes" / "newer.md").write_text("Another note.\n", encoding="utf-8")
    code, out, err = run(capsys, "index", "-q", str(corpus))
    assert (code, out, err) == (0, "", "")


def test_index_of_documents_only(env, tmp_path: Path, capsys) -> None:
    folder = tmp_path / "docs"
    add_documents(folder)
    _, _, err = run(capsys, "index", str(folder))
    assert "No files to index" in err and LINE in err
    _, _, err = run(capsys, "index", str(folder / "a.docx"))
    assert "Skipped" in err and ".docx files are not supported yet." in err
    assert "--ext" not in err and LINE not in err


def test_search_says_it_only_when_indexing_new_files(env, corpus: Path, capsys) -> None:
    add_documents(corpus / "office")
    code, out, err = run(capsys, "bread", str(corpus), "-l")
    assert code == 0 and LINE in err and LINE not in out
    first = out
    # The next search finds nothing new to index: no line, same output.
    code, out, err = run(capsys, "bread", str(corpus), "-l")
    assert code == 0 and LINE not in err and out == first
    (corpus / "notes" / "new.md").write_text("A new note.\n", encoding="utf-8")
    code, out, err = run(capsys, "bread", str(corpus), "--json")
    assert code == 0 and LINE in err and LINE not in out
    (corpus / "notes" / "newer.md").write_text("Another note.\n", encoding="utf-8")
    code, out, err = run(capsys, "bread", str(corpus), "-q")
    assert code == 0 and out == "" and LINE not in err


def test_search_of_documents_only_folder(env, tmp_path: Path, capsys) -> None:
    folder = tmp_path / "docs"
    add_documents(folder)
    code, out, err = run(capsys, "report", str(folder))
    assert code == 2 and out == "" and LINE in err


@pytest.fixture
def idx(tmp_path: Path, env, monkeypatch):
    monkeypatch.setattr(embedder, "embedder_factory", lambda threads, quiet=True: lambda v: env)
    with anygrep.Index(tmp_path / "api.sqlite", threads=2) as i:
        yield i


def test_update_result_counts_documents(idx: anygrep.Index, corpus: Path, capfd) -> None:
    add_documents(corpus / "office")
    res = idx.update(corpus)
    assert res.new > 0 and res.unsupported_documents == 5
    assert idx.update(corpus).unsupported_documents == 5  # every scan counts them
    out, err = capfd.readouterr()
    assert out == "" and LINE not in err
