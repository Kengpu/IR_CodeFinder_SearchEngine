"""Query parsing for lexical and Boolean retrieval (LEX-01/LEX-05)."""
from __future__ import annotations

from dataclasses import dataclass

from ..indexing.tokenizer import canonical_token, tokenize


@dataclass(frozen=True)
class QueryToken:
    value: str
    kind: str  # TERM, AND, OR, NOT, LPAREN, RPAREN


@dataclass
class ParsedQuery:
    raw: str
    tokens: list[QueryToken]

    @property
    def terms(self) -> list[str]:
        return [t.value for t in self.tokens if t.kind == "TERM"]

    @property
    def operators(self) -> list[str]:
        return [t.kind for t in self.tokens if t.kind in {"AND", "OR", "NOT"}]


_BOOLEAN_WORDS = {"AND", "OR", "NOT"}


def parse_query(query: str) -> ParsedQuery:
    """Tokenize a query while recognizing Boolean operators.

    Implicit whitespace between terms is treated as OR for user-friendly
    natural-language queries. Explicit operators are retained for Boolean
    search. Parentheses are emitted by the identifier-preserving tokenizer.
    """
    raw_tokens = tokenize(query, remove_stopwords=False, stemming=False)
    tokens: list[QueryToken] = []
    for raw in raw_tokens:
        upper = raw.upper()
        if raw in _BOOLEAN_WORDS:
            tokens.append(QueryToken(upper, upper))
        elif raw == "(":
            tokens.append(QueryToken(raw, "LPAREN"))
        elif raw == ")":
            tokens.append(QueryToken(raw, "RPAREN"))
        else:
            tokens.append(QueryToken(canonical_token(raw), "TERM"))
    return ParsedQuery(query, tokens)


def _insert_implicit_or(tokens: list[QueryToken]) -> list[QueryToken]:
    """Insert OR between adjacent operands, but not around explicit NOT."""
    out: list[QueryToken] = []
    left_ends = {"TERM", "RPAREN"}
    right_starts = {"TERM", "LPAREN", "NOT"}
    for token in tokens:
        if out and out[-1].kind in left_ends and token.kind in right_starts:
            out.append(QueryToken("OR", "OR"))
        out.append(token)
    return out


def to_postfix(parsed: ParsedQuery) -> list[QueryToken]:
    """Convert Boolean expression to postfix using NOT > AND > OR precedence."""
    precedence = {"OR": 1, "AND": 2, "NOT": 3}
    output: list[QueryToken] = []
    stack: list[QueryToken] = []
    for token in _insert_implicit_or(parsed.tokens):
        if token.kind == "TERM":
            output.append(token)
        elif token.kind == "LPAREN":
            stack.append(token)
        elif token.kind == "RPAREN":
            while stack and stack[-1].kind != "LPAREN":
                output.append(stack.pop())
            if stack and stack[-1].kind == "LPAREN":
                stack.pop()
        else:
            while (stack and stack[-1].kind != "LPAREN"
                   and precedence.get(stack[-1].kind, 0) >= precedence[token.kind]):
                output.append(stack.pop())
            stack.append(token)
    while stack:
        if stack[-1].kind != "LPAREN":
            output.append(stack.pop())
        else:
            stack.pop()
    return output


def query_terms_for_highlight(query: str) -> list[str]:
    """Return non-operator terms, including code punctuation, for highlighting."""
    return [token.value for token in parse_query(query).tokens if token.kind == "TERM"]
