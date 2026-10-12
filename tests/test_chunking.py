from __future__ import annotations

import itertools
import re

from anygrep.chunking import chunk_text


def ws_offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def ntok(s: str) -> int:
    return len(ws_offsets(s))


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("", ws_offsets) == []
    assert chunk_text("  \n\n ", ws_offsets) == []


def test_short_text_is_one_chunk_with_lines() -> None:
    text = "alpha beta\ngamma\n\ndelta"
    (c,) = chunk_text(text, ws_offsets, max_tokens=50, overlap=5)
    assert c.text == text
    assert (c.line_start, c.line_end) == (1, 4)
    assert (c.char_start, c.char_end) == (0, len(text))


def test_chunks_respect_budget_and_overlap() -> None:
    lines = [f"line{i} " + " ".join(f"w{i}_{j}" for j in range(9)) for i in range(40)]
    text = "\n".join(lines)
    chunks = chunk_text(text, ws_offsets, max_tokens=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert ntok(c.text) <= 50
        assert text[c.char_start : c.char_end] == c.text
        assert c.text.splitlines()[0] == lines[c.line_start - 1]
    # consecutive chunks overlap by at least one line and always advance
    for a, b in itertools.pairwise(chunks):
        assert b.line_start <= a.line_end
        assert b.line_start > a.line_start
    assert chunks[-1].line_end == 40


def test_overlong_line_is_split() -> None:
    text = " ".join(f"t{i}" for i in range(250))
    chunks = chunk_text(text, ws_offsets, max_tokens=100, overlap=20)
    assert len(chunks) == 3
    assert all(ntok(c.text) <= 100 for c in chunks)
    assert all(c.line_start == c.line_end == 1 for c in chunks)
    assert chunks[0].text.startswith("t0 ")
    assert chunks[-1].text.endswith("t249")


def test_heading_starts_new_chunk_when_half_full() -> None:
    body = "\n".join("word " * 8 for _ in range(4))  # ~32 tokens
    text = body + "\n# Next section\n" + body
    chunks = chunk_text(text, ws_offsets, max_tokens=60, overlap=0)
    assert any(c.text.startswith("# Next section") for c in chunks)


def nl_offsets(text: str) -> list[tuple[int, int]]:
    """Like a real tokenizer: words and runs of newlines are tokens."""
    return [m.span() for m in re.finditer(r"\S+|\n+", text)]


def tokens_in(text: str) -> int:
    return len(nl_offsets(text))


def test_newline_tokens_are_counted() -> None:
    cases = [
        "\n".join(f"- item{i}" for i in range(400)),
        "\n\n".join(f"word{i}" for i in range(400)),
        ("x" + "\n" * 30) * 200,
    ]
    for text in cases:
        chunks = chunk_text(text, nl_offsets, max_tokens=50, overlap=10)
        assert chunks
        for c in chunks:
            assert tokens_in(c.text) <= 50
        assert text[chunks[-1].char_start : chunks[-1].char_end].strip() == chunks[-1].text.strip()
        assert chunks[-1].char_end >= len(text.rstrip("\n"))


def test_long_paragraph_lines_still_overlap() -> None:
    paras = [" ".join(f"p{i}w{j}" for j in range(40)) for i in range(12)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, ws_offsets, max_tokens=100, overlap=10)
    assert len(chunks) > 3
    for a, b in itertools.pairwise(chunks):
        assert a.char_start < b.char_start < a.char_end  # shares text, still advances
        assert ntok(text[b.char_start : a.char_end]) <= 10
        assert ntok(b.text) <= 100
        assert text[b.char_start : b.char_end] == b.text


def test_huge_text_is_tokenized_in_blocks(monkeypatch) -> None:
    from anygrep import chunking

    lines = [" ".join(f"w{i}x{j}" for j in range(i % 13)) for i in range(3000)]
    text = "\n".join(lines) + "\n\n\n" + " ".join(f"long{i}" for i in range(4000))
    whole = chunk_text(text, ws_offsets, max_tokens=40, overlap=8)
    seen: list[int] = []

    def offsets(s: str) -> list[tuple[int, int]]:
        seen.append(len(s))
        return ws_offsets(s)

    monkeypatch.setattr(chunking, "TOKENIZE_BLOCK_CHARS", 1000)
    blocks = chunk_text(text, offsets, max_tokens=40, overlap=8)
    assert len(seen) > 50 and max(seen) <= 1000
    assert blocks == whole  # cut at whitespace: the same tokens, the same chunks


def test_huge_text_is_chunked_in_blocks_of_lines(monkeypatch) -> None:
    from anygrep import chunking

    lines = [" ".join(f"w{i}x{j}" for j in range(i % 7)) for i in range(4000)]
    text = "\n".join(lines) + "\n" + " ".join(f"long{i}" for i in range(3000)) + "\nend"
    monkeypatch.setattr(chunking, "CHUNK_BLOCK_CHARS", 2000)
    chunks = chunk_text(text, ws_offsets, max_tokens=40, overlap=8)
    assert len(chunks) > 100
    all_lines = text.split("\n")
    for c in chunks:
        # Offsets and line numbers refer to the whole text.
        assert text[c.char_start : c.char_end] == c.text
        assert c.text.split("\n")[0] in all_lines[c.line_start - 1]
        assert text.count("\n", 0, c.char_start) == c.line_start - 1
        assert ntok(c.text) <= 40
    # Every word is in some chunk.
    words = set(re.findall(r"\S+", text))
    assert words == set(itertools.chain.from_iterable(re.findall(r"\S+", c.text) for c in chunks))


def test_runs_of_blank_lines_stay_in_bounded_blocks(monkeypatch) -> None:
    from anygrep import chunking

    seen: list[int] = []

    def offsets(s: str) -> list[tuple[int, int]]:
        seen.append(len(s))
        return ws_offsets(s)

    monkeypatch.setattr(chunking, "TOKENIZE_BLOCK_CHARS", 1000)
    text = "a\n" + "\n" * 50_000 + "b"
    chunks = chunk_text(text, offsets, max_tokens=40, overlap=8)
    assert max(seen) <= 2000
    assert chunks[0].line_start == 1 and chunks[-1].line_end == 50_002
    assert chunks[-1].text.endswith("b")
