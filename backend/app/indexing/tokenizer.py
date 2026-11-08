"""Identifier-preserving text normalization and tokenization (IDX-08/09).

Programming tokens must remain searchable: exception names, error codes,
operators, file paths, commands, and common punctuation are kept intact.
Natural-language words can be lowercased, stop-filtered, and stemmed via
independent switches for ablation experiments.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

from .. import config

# Order matters: longer/special forms are matched before generic identifiers.
_TOKEN_RE = re.compile(
    r"""(?x)
    (?P<winpath>[A-Za-z]:\\(?:[^\s<>|\"*?]+(?:\\[^\s<>|\"*?]+)*))
  | (?P<unixpath>/(?:[A-Za-z0-9_.@+-]+/)*[A-Za-z0-9_.@+-]+)
  | (?P<command>(?:npm|npx|yarn|pnpm|pip|python|git|docker|kubectl)\s+(?:install|i|add|run|start|build|test|update|upgrade|remove|uninstall|commit|push|pull|checkout|exec|apply|delete|version))
  | (?P<errorcode>(?:HTTP\s*)?\d{3,5}(?:[-_][A-Za-z0-9]+)*)
  | (?P<operator>===|!==|==|!=|<=|>=|=>|->|::|:=|\+=|-=|\*=|/=|&&|\|\||\?\?|\.\.\.|\.\.|
      \+\+|--|<<|>>|\*\*|//)
  | (?P<identifier>[A-Za-z_$][A-Za-z0-9_$]*(?:(?:\.|-)[A-Za-z0-9_$]+)*)
  | (?P<number>\d+(?:\.\d+)*(?:[eE][+-]?\d+)?)
  | (?P<punct>[(){}\[\],;:.!?<>/=+*%&|^~`#@\\])
  | (?P<other>\S)
    """
)

# Keep these as tokens even when they appear alone; useful in code/error queries.
_PRESERVED_PUNCT = set("(){}[],:;.!?<>/=+*%&|^~`#@\\")

# Compact English stop list; only applied to plain alphabetic words.
_STOPWORDS = frozenset("""
a an the and or but if then else when while where what which who whom whose
how why is are was were be been being do does did have has had having can could
may might must shall should will would of at by for with about against between
into through during before after above below to from up down in out on off over
under again further once here there all any both each few more most other some
such no nor not only own same so than too very s t don should now as it its this
that these those i me my we our you your he she they them their am were been
being because until while as per via also just than
""".split())


def normalize_text(text: str, *, lowercase: bool | None = None,
                   accent_folding: bool | None = None) -> str:
    """Unicode-normalize text while retaining source-code characters."""
    if not text:
        return ""
    if accent_folding is None:
        accent_folding = config.ENABLE_ACCENT_FOLDING
    # NFKC normalizes compatibility forms but does not remove code punctuation.
    text = unicodedata.normalize("NFKC", text)
    if accent_folding:
        text = "".join(c for c in unicodedata.normalize("NFKD", text)
                       if not unicodedata.combining(c))
    # Normalize typographic quotes/dashes commonly pasted into questions.
    text = text.translate(str.maketrans({
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00a0": " ",
    }))
    if lowercase is True:
        text = text.lower()
    return text


def _is_identifier(token: str) -> bool:
    """True for code-like terms that should not be stop-filtered or stemmed.

    A leading capital alone is normal prose (``Python``/``Java``), while an
    internal capital is a useful signal for an identifier (``TypeError`` or
    ``MyClass``). Case-insensitive indexing still makes both forms searchable.
    """
    return (any(ch.isupper() for ch in token[1:])
            or any(ch.isdigit() for ch in token)
            or any(ch in "_$.-/\\" for ch in token)
            or token.startswith(("http", "err_")))


@lru_cache(maxsize=1)
def _porter_stemmer():
    """Load nltk's Porter stemmer only when stemming is requested."""
    try:
        from nltk.stem import PorterStemmer
        return PorterStemmer()
    except ImportError as exc:
        raise RuntimeError("Install nltk to enable stemming: pip install nltk") from exc


def tokenize(text: str, *, remove_stopwords: bool | None = None,
             stemming: bool | None = None, lowercase: bool | None = None) -> list[str]:
    """Tokenize prose and source code without destroying useful syntax.

    Natural-language tokens are case-folded by default. Code-like identifiers
    preserve their original spelling (e.g. TypeError, MyClass, C:\\path),
    while indexers case-fold keys for case-insensitive lookup. Operators,
    punctuation, error codes, and commands are separate stable tokens.
    """
    if remove_stopwords is None:
        remove_stopwords = config.ENABLE_STOPWORDS
    if stemming is None:
        stemming = config.ENABLE_STEMMING
    # The tokenizer intentionally preserves case in source; lowercase here is
    # a compatibility switch for experiments that want a fully lowercased view.
    if lowercase is None:
        lowercase = False
    text = normalize_text(text, lowercase=False)
    tokens: list[str] = []
    stemmer = _porter_stemmer() if stemming else None
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if not token or token.isspace():
            continue
        # Multiword commands are preserved as phrase token and split into
        # component tokens as well, supporting both exact and broad searches.
        if match.lastgroup == "command":
            parts = token.split()
            tokens.extend(parts)
            tokens.append("_".join(parts))
            continue
        # Ignore unsupported isolated symbols but preserve operators/punct.
        if match.lastgroup == "other":
            continue
        code_like = _is_identifier(token) or match.lastgroup in {
            "winpath", "unixpath", "errorcode", "operator", "punct", "number"
        }
        if lowercase:
            token = token.lower()
        elif not code_like:
            token = token.lower()
        # Stop words only affect plain English alphabetic tokens.
        if (remove_stopwords and token.lower() in _STOPWORDS
                and match.lastgroup == "identifier" and not code_like):
            continue
        # Stemming applies only to ordinary alphabetic prose words.
        if stemming and match.lastgroup == "identifier" and not code_like \
                and token.isalpha():
            token = stemmer.stem(token)
        if token:
            tokens.append(token)
    return tokens


def canonical_token(token: str) -> str:
    """Case-insensitive index key that does not alter token boundaries."""
    return token.casefold()


def query_terms(text: str, **kwargs) -> list[str]:
    """Tokenize a query, excluding Boolean operators from the term list."""
    return [t for t in tokenize(text, **kwargs)
            if t.upper() not in {"AND", "OR", "NOT"}]


def identifier_tokens(text: str) -> list[str]:
    """Convenience helper used by preservation tests and diagnostics."""
    return [t for t in tokenize(text, remove_stopwords=False)
            if _is_identifier(t) or t in _PRESERVED_PUNCT]


def tokenize_many(texts: Iterable[str], **kwargs) -> list[list[str]]:
    return [tokenize(t, **kwargs) for t in texts]
