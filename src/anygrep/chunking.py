"""Token-aware text chunking that keeps line numbers and character offsets.

Text is split into lines; lines are packed greedily into chunks of at most
``max_tokens`` model tokens, with roughly ``overlap`` tokens of trailing
context repeated at the start of the next chunk. Lines longer than a chunk
are cut at token boundaries. A Markdown heading starts a new chunk when the
current one is already reasonably full.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Text is tokenized in blocks of about this many characters, cut after a line
# break where possible, so a huge file never needs one tokenizer call (and one
# list of offsets) for all of it.
TOKENIZE_BLOCK_CHARS = 1 << 16
# Text longer than this is chunked in blocks of about this many characters,
# cut after a line break, so the per-line bookkeeping of a huge file (millions
# of short or blank lines) stays small. Chunks do not overlap across blocks.
CHUNK_BLOCK_CHARS = 1 << 20


@dataclass(frozen=True)
class Chunk:
    text: str
    char_start: int
    char_end: int
    line_start: int  # 1-based, inclusive
    line_end: int  # 1-based, inclusive


@dataclass(frozen=True)
class _Seg:
    start: int
    end: int
    tokens: int
    line: int  # 0-based line index


def chunk_text(
    text: str,
    offsets_fn: Callable[[str], list[tuple[int, int]]],
    max_tokens: int = 384,
    overlap: int = 48,
) -> list[Chunk]:
    if len(text) <= CHUNK_BLOCK_CHARS:
        return _chunk_block(text, offsets_fn, max_tokens, overlap)
    chunks: list[Chunk] = []
    pos, line, n = 0, 0, len(text)
    while pos < n:
        end = min(pos + CHUNK_BLOCK_CHARS, n)
        if end < n:
            cut = text.rfind("\n", pos, end)
            if cut < pos:  # one very long line: up to its end
                cut = text.find("\n", end)
            end = n if cut < 0 else cut + 1
        block = text[pos:end]
        for c in _chunk_block(block, offsets_fn, max_tokens, overlap):
            chunks.append(
                Chunk(c.text, c.char_start + pos, c.char_end + pos,
                      c.line_start + line, c.line_end + line)
            )  # fmt: skip
        line += block.count("\n")
        pos = end
    return chunks


def _chunk_block(
    text: str,
    offsets_fn: Callable[[str], list[tuple[int, int]]],
    max_tokens: int,
    overlap: int,
) -> list[Chunk]:
    if not text.strip():
        return []
    overlap = max(0, min(overlap, max_tokens // 2))
    tok_starts = token_starts(text, offsets_fn)

    # Line boundaries. A line's tokens are those starting between its first
    # character and the start of the next line, so newline tokens count too.
    line_starts = [0]
    i = text.find("\n")
    while i >= 0:
        line_starts.append(i + 1)
        i = text.find("\n", i + 1)
    line_ends = [*(s - 1 for s in line_starts[1:]), len(text)]
    line_next = [*line_starts[1:], len(text)]

    def tok_index(pos: int) -> int:
        return int(np.searchsorted(tok_starts, pos, "left"))

    segs: list[_Seg] = []
    for li, (a, b, nxt) in enumerate(zip(line_starts, line_ends, line_next, strict=True)):
        i0, i1 = tok_index(a), tok_index(nxt) if nxt > b else len(tok_starts)
        n = i1 - i0
        if n <= max_tokens:
            segs.append(_Seg(a, b, n, li))
            continue
        # Split an overlong line at token boundaries.
        step = max_tokens - overlap
        k = i0
        while k < i1:
            kend = min(k + max_tokens, i1)
            s = a if k == i0 else min(int(tok_starts[k]), b)
            e = b if kend >= i1 else min(int(tok_starts[kend]), b)
            segs.append(_Seg(s, e, kend - k, li))
            if kend >= i1:
                break
            k += step

    def tail(seg: _Seg, want: int) -> _Seg | None:
        """The last `want` tokens of a segment, cut at a token boundary."""
        t0 = tok_index(seg.start)
        t1 = t0 + seg.tokens
        k = max(t0 + 1, t1 - want)
        if k >= t1 or tok_starts[k] >= seg.end:
            return None
        return _Seg(int(tok_starts[k]), seg.end, t1 - k, seg.line)

    chunks: list[Chunk] = []
    i = 0
    n = len(segs)
    carry: _Seg | None = None  # partial overlap from the previous chunk
    while i < n:
        if carry is not None and carry.tokens + segs[i].tokens > max_tokens:
            carry = None
        parts: list[_Seg] = [carry] if carry is not None else []
        total = carry.tokens if carry is not None else 0
        j = i
        while j < n:
            t = segs[j].tokens
            if j > i and total + t > max_tokens:
                break
            if j > i and total >= max_tokens // 2 and text.startswith("#", segs[j].start):
                break
            parts.append(segs[j])
            total += t
            j += 1
        while len(parts) > 1 and parts[0].start == parts[0].end:
            parts.pop(0)  # do not start a chunk with blank lines
        a, b = parts[0].start, parts[-1].end
        body = text[a:b]
        if body.strip():
            chunks.append(Chunk(body, a, b, parts[0].line + 1, parts[-1].line + 1))
        if j >= n:
            break
        # Step back to include ~overlap tokens of context, but always advance.
        k = j
        back = 0
        while k - 1 > i and back + segs[k - 1].tokens <= overlap:
            k -= 1
            back += segs[k].tokens
        carry = None
        if back < overlap and k - 1 >= i and segs[k - 1].tokens > overlap - back:
            # The previous line is too long to repeat whole: repeat its end.
            carry = tail(segs[k - 1], overlap - back)
        i = k
    return chunks


def token_starts(text: str, offsets_fn: Callable[[str], list[tuple[int, int]]]) -> np.ndarray:
    """Start offsets of the tokens of text, tokenized block by block."""
    parts: list[np.ndarray] = []
    pos, n = 0, len(text)
    while pos < n:
        end = min(pos + TOKENIZE_BLOCK_CHARS, n)
        if end < n:
            cut = text.rfind("\n", pos, end)
            if cut > pos:
                end = cut + 1
                # Keep a run of blank lines together, up to another block.
                limit = min(n, end + TOKENIZE_BLOCK_CHARS)
                while end < limit and text[end] == "\n":
                    end += 1
            else:  # one very long line: cut before a space, else anywhere
                cut = text.rfind(" ", pos + 1, end)
                if cut > pos:
                    end = cut
        offs = offsets_fn(text[pos:end])
        parts.append(np.fromiter((s for s, _ in offs), np.int64, len(offs)) + pos)
        pos = end
    return np.concatenate(parts) if parts else np.zeros(0, np.int64)
