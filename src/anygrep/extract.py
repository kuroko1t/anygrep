"""File type detection and content extraction (text, PDF pages)."""

from __future__ import annotations

import html
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

TEXT_EXTS = frozenset(
    {
        ".txt", ".text", ".md", ".markdown", ".mdx", ".rst", ".org", ".adoc",
        ".asciidoc", ".tex", ".wiki", ".html", ".htm",
    }
)  # fmt: skip
PDF_EXTS = frozenset({".pdf"})
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"})

REPLACEMENT_CHAR = chr(0xFFFD)

KINDS = ("text", "pdf", "image")

# Default size limits in bytes.
MAX_SIZE = {"text": 20 * 2**20, "pdf": 300 * 2**20, "image": 80 * 2**20}


def kind_of(path: Path, extra_text_exts: frozenset[str] = frozenset()) -> str | None:
    ext = path.suffix.lower()
    if ext in TEXT_EXTS or ext in extra_text_exts:
        return "text"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    return None


class SkipFile(Exception):
    """Raised when a file should be recorded as skipped (not an error)."""


def looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def decode_text(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if looks_binary(data):
        raise SkipFile("binary content")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    text = data.decode("utf-8", errors="replace")
    bad = text.count(REPLACEMENT_CHAR)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if bad > 0.3 * non_ascii:
        # Mostly invalid UTF-8: a legacy single-byte encoding, most often cp1252.
        return data.decode("cp1252", errors="replace")
    return text


_RAW_OPEN = re.compile(r"<(script|style)\b", re.I)
_RAW_CLOSE = {
    "script": re.compile(r"</script\s*>", re.I),
    "style": re.compile(r"</style\s*>", re.I),
}
_TAG_NAME = re.compile(r"</?\s*([a-z][\w:-]*)\b", re.I)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
)


# A tag up to its closing '>': a quote right after '=' starts an attribute value
# that may contain '>'; any other quote is a plain character, as in browsers.
# The alternatives start with distinct characters, so matching stays linear.
_TAG = re.compile(r"""<[^>"'=]*(?:(?:=\s*"[^"]*"|=\s*'[^']*'|=(?!\s*["'])|["'])[^>"'=]*)*>""")


def _tag_end(s: str, start: int) -> int:
    """Find the end of the tag at s[start] ('<'), or -1 if no '>' follows."""
    if s.startswith("<!--", start):
        close = s.find("-->", start + 4)
        if close >= 0:
            return close + 3
    gt = s.find(">", start + 1)
    if gt < 0:
        return -1
    if not s.startswith(("<!", "<?"), start):
        m = _TAG.match(s, start)
        if m:
            return m.end()
    return gt + 1  # declaration, or an unbalanced quote: end at the first '>'


def html_to_text(s: str) -> str:
    """Drop tags, scripts and styles; keep line structure so line numbers stay
    roughly meaningful. One pass over the text: a tag or script that is never
    closed cannot make it scan to the end of the file again for every '<'."""
    out: list[str] = []
    pos = 0
    # Where a closing tag was last searched for and not found: none after it.
    no_close = {"script": len(s) + 1, "style": len(s) + 1}
    while True:
        lt = s.find("<", pos)
        if lt < 0:
            break
        end = -1
        m = _RAW_OPEN.match(s, lt)
        if m:
            name = m.group(1).lower()
            if lt < no_close[name]:
                c = _RAW_CLOSE[name].search(s, m.end())
                if c:
                    end = c.end()
                else:
                    no_close[name] = lt
        if end < 0:
            end = _tag_end(s, lt)
            if end < 0:
                break  # no tag can end after this point: the rest is text
        if end == lt + 2:  # "<>" is not a tag
            out.append(s[pos:end])
        else:
            out.append(s[pos:lt])
            newlines = s.count("\n", lt, end)
            if newlines:
                out.append("\n" * newlines)
            else:
                tag = _TAG_NAME.match(s, lt)
                if tag and tag.group(1).lower() in _BLOCK_TAGS:
                    out.append(" ")
        pos = end
    out.append(s[pos:])
    return html.unescape("".join(out))


def read_text_file(path: Path) -> str:
    text = decode_text(path.read_bytes())
    if path.suffix.lower() in (".html", ".htm"):
        text = html_to_text(text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass
class PdfPage:
    number: int  # 1-based
    text: str


def read_pdf_pages(path: Path) -> Iterator[PdfPage]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as e:
        msg = str(e).lower()
        if "password" in msg:
            raise SkipFile("encrypted PDF") from e
        raise
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            try:
                tp = page.get_textpage()
                try:
                    text = tp.get_text_range()
                finally:
                    tp.close()
            finally:
                page.close()
            text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
            yield PdfPage(i + 1, text)
    finally:
        pdf.close()
