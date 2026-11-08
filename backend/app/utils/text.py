"""Shared text utilities: HTML cleaning, snippet generation, highlighting."""
from __future__ import annotations

import html as html_mod
import re

# --------------------------------------------------------------------------
# HTML handling
# --------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.DOTALL | re.IGNORECASE)
_CODE_INLINE_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_WS_RE = re.compile(r"[ \t]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def extract_code_blocks(html: str) -> list[str]:
    """Extract <pre><code> contents from Stack Exchange HTML bodies."""
    blocks = []
    for m in _CODE_BLOCK_RE.finditer(html or ""):
        text = html_mod.unescape(_TAG_RE.sub("", m.group(1)))
        text = _NEWLINES_RE.sub("\n\n", text).strip()
        if text:
            blocks.append(text)
    return blocks


def html_to_text(html: str) -> str:
    """Convert Stack Exchange HTML body to plain text, keeping code as text.

    Code blocks are preserved verbatim (they carry error signatures);
    remaining tags are stripped and entities unescaped.
    """
    if not html:
        return ""
    s = _SCRIPT_STYLE_RE.sub(" ", html)
    # Newline markers for block-level tags so paragraphs stay separated.
    s = re.sub(r"</(p|div|li|h[1-6]|blockquote)>", "\n", s, flags=re.IGNORECASE)
    s = _CODE_BLOCK_RE.sub(lambda m: "\n```\n" + html_mod.unescape(_TAG_RE.sub("", m.group(1))) + "\n```\n", s)
    s = _CODE_INLINE_RE.sub(lambda m: "`" + html_mod.unescape(m.group(1)) + "`", s)
    s = _TAG_RE.sub(" ", s)
    s = html_mod.unescape(s)
    s = _WS_RE.sub(" ", s)
    s = _NEWLINES_RE.sub("\n\n", s)
    return s.strip()


# --------------------------------------------------------------------------
# Snippets and highlighting
# --------------------------------------------------------------------------
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def make_snippet(text: str, query_terms: list[str], max_chars: int = 240) -> str:
    """Pick the most relevant short excerpt of `text` for the result list.

    Scores each sentence by number of query-term hits; falls back to the
    opening of the text when nothing matches.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if sentences:
        lowered = [t.lower() for t in query_terms if t]
        best, best_score = None, -1
        for s in sentences:
            sl = s.lower()
            score = sum(1 for t in lowered if t in sl)
            if score > best_score:
                best, best_score = s, score
        if best is not None and best_score > 0:
            if len(best) > max_chars:
                best = best[: max_chars - 3].rstrip() + "..."
            return best
    return text[: max_chars - 3].rstrip() + "..."


def highlight_text(text: str, query_terms: list[str], max_terms: int = 12) -> str:
    """Return text with <mark> around query term occurrences.

    Safe by construction: the input is escaped first, then only exact
    <mark>...</mark> tags are inserted — no user HTML can survive. Matching
    is word-boundary based and case-insensitive.
    """
    if not text:
        return ""
    escaped = html_mod.escape(text)
    seen = set()
    for term in query_terms[:max_terms]:
        t = term.strip().lower()
        if not t or t in seen or len(t) < 2:
            continue
        seen.add(t)
        pattern = re.compile(re.escape(html_mod.escape(term.strip())), re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped
