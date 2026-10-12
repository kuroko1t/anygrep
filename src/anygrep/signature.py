"""Word signatures for telling near-duplicate files from template documents.

A signature is a bottom-k sketch: the SIG_SIZE smallest 32-bit hashes of the
file's word pairs (Japanese, Chinese and Korean count per character). It is
computed once at index time, so comparing two files at search time costs a
few microseconds whatever their size.

`changed_share` estimates the share of word pairs of one file that the other
lacks, from the side that is better contained in the other, so lines added
to or removed from a copy do not count. A replaced word changes two pairs:
monthly payslips or statements (14 to 36 percent of words replaced) differ
in well over a third of their pairs, edited copies of a note in a few.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

import numpy as np

SIG_SIZE = 256
# Text is split into words in blocks of about this many characters.
_BLOCK_CHARS = 1 << 20

_WORD = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]|\w+")
_MUL1 = np.uint64(0x9E3779B97F4A7C15)
_MUL2 = np.uint64(0xBF58476D1CE4E5B9)
_MUL3 = np.uint64(0x94D049BB133111EB)


def _mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hash of the pairs (a[i], b[i]) of 64-bit word hashes, as uint32."""
    with np.errstate(over="ignore"):
        x = a * _MUL1 + b
        x ^= x >> np.uint64(30)
        x *= _MUL2
        x ^= x >> np.uint64(27)
        x *= _MUL3
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(32)).astype(np.uint32)


class SignatureBuilder:
    """Feed text in order with update(); finish() returns the signature."""

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None  # hash of the last word so far
        self._sketch = np.zeros(0, np.uint32)
        self._hashes: dict[str, int] = {}

    def _hash(self, w: str) -> int:
        h = self._hashes.get(w)
        if h is None:
            h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "little")
            if len(self._hashes) < 1_000_000:
                self._hashes[w] = h
        return h

    def update(self, text: str) -> None:
        pos, n = 0, len(text)
        while pos < n:
            end = min(pos + _BLOCK_CHARS, n)
            if end < n:
                cut = max(text.rfind(" ", pos, end), text.rfind("\n", pos, end))
                if cut > pos:
                    end = cut
            self._add(text[pos:end])
            pos = end

    def _add(self, text: str) -> None:
        words = _WORD.findall(text.lower())
        if not words:
            return
        h = np.fromiter((self._hash(w) for w in words), np.uint64, len(words))
        if self._prev is not None:
            h = np.concatenate([self._prev, h])
        self._prev = h[-1:]
        if len(h) >= 2:
            pairs = _mix(h[:-1], h[1:])
            self._sketch = np.unique(np.concatenate([self._sketch, pairs]))[:SIG_SIZE]

    def finish(self) -> bytes | None:
        if not len(self._sketch):
            if self._prev is None:
                return None
            self._sketch = _mix(self._prev, self._prev)  # a single word
        return self._sketch.astype("<u4").tobytes()


def signature(texts: Iterable[str]) -> bytes | None:
    """The signature of the concatenated texts (None without words)."""
    b = SignatureBuilder()
    for t in texts:
        b.update(t)
    return b.finish()


def changed_share(a: bytes | None, b: bytes | None) -> float | None:
    """Estimated share of word pairs of the better-contained file that the
    other file lacks (0.0: one contains the other). None without a signature."""
    if not a or not b:
        return None
    sa = np.frombuffer(a, "<u4")
    sb = np.frombuffer(b, "<u4")
    best = 0.0
    for x, y in ((sa, sb), (sb, sa)):
        # y holds every hash of its file up to its largest one, so membership
        # is exact for the hashes of x in that range.
        if len(y) >= SIG_SIZE:
            x = x[x <= y[-1]]
        if len(x):
            best = max(best, float(np.isin(x, y, assume_unique=True).mean()))
    return 1.0 - best
