"""Lexical retrieval: TF-IDF, BM25, and Boolean search (LEX-02..09).

Both ranked algorithms are implemented directly against the project's
inverted index. TF-IDF uses log-scaled TF and smoothed IDF; BM25 uses the
standard Robertson-Sparck Jones IDF with configurable k1/b. Scores are
un-normalized internally and min-max normalized by SearchPipeline for fusion.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .. import config
from ..indexing.inverted import InvertedIndex
from ..indexing.tokenizer import canonical_token, tokenize
from ..utils.text import highlight_text, make_snippet
from .query import parse_query, query_terms_for_highlight, to_postfix


@dataclass
class SearchHit:
    doc_id: str
    score: float
    explanation: dict[str, Any]
    snippet: str = ""


class LexicalSearcher:
    """Reusable lexical searcher with TF-IDF, BM25, Boolean and PRF hook."""

    def __init__(self, index: InvertedIndex, *, k1: float = config.BM25_K1,
                 b: float = config.BM25_B,
                 tfidf_smoothing: float = config.TFIDF_SMOOTHING):
        self.index = index
        self.k1 = k1
        self.b = b
        self.tfidf_smoothing = tfidf_smoothing

    @property
    def document_count(self) -> int:
        return self.index.total_docs

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------
    def _terms(self, query: str) -> list[str]:
        return [canonical_token(t) for t in tokenize(
            query, remove_stopwords=False, stemming=False)
                if t.upper() not in {"AND", "OR", "NOT"} and t not in {"(", ")"}]

    def _allowed(self, language: str | None, technology: str | None,
                 allowed_ids: set[str] | None) -> set[str] | None:
        meta_ids = self.index.filtered_doc_ids(language=language, technology=technology)
        if meta_ids is None:
            return allowed_ids
        if allowed_ids is None:
            return meta_ids
        return meta_ids & allowed_ids

    def _make_hit(self, doc_id: str, score: float, query: str,
                  terms: list[str], explanation: dict) -> SearchHit:
        doc = self.index.documents[doc_id]
        snippet_source = doc.get("error_message") or doc.get("description", "")
        snippet = make_snippet(snippet_source, terms)
        # The raw snippet is kept as text in API payload; frontend highlights
        # safely using the explicit highlight_terms field.
        return SearchHit(doc_id, float(score), explanation, snippet)

    # ------------------------------------------------------------------
    # TF-IDF (from scratch)
    # ------------------------------------------------------------------
    def tfidf(self, query: str, *, top_k: int = 10,
              language: str | None = None, technology: str | None = None,
              allowed_ids: set[str] | None = None) -> list[SearchHit]:
        terms = self._terms(query)
        if not terms or self.document_count == 0:
            return []
        allowed = self._allowed(language, technology, allowed_ids)
        qtf = Counter(terms)
        scores: dict[str, float] = {}
        term_contrib: dict[str, dict[str, float]] = {}
        n = self.document_count
        for term, qcount in qtf.items():
            postings = self.index.postings.get(term, {})
            df = len(postings)
            if df == 0:
                continue
            # Smoothed IDF avoids zero/negative values for ubiquitous terms.
            idf = math.log(1.0 + (n - df + self.tfidf_smoothing) /
                           (df + self.tfidf_smoothing))
            for doc_id, posting in postings.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                tf_weight = 1.0 + math.log(posting.tf) if posting.tf > 0 else 0.0
                contribution = tf_weight * idf * qcount
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution
                term_contrib.setdefault(doc_id, {})[term] = contribution
        hits = []
        for doc_id, score in scores.items():
            if score <= 0:
                continue
            contrib = term_contrib.get(doc_id, {})
            hits.append(self._make_hit(doc_id, score, query, terms, {
                "method": "tfidf",
                "matched_terms": list(contrib),
                "term_contributions": {k: round(v, 6) for k, v in contrib.items()},
                "reason": f"Matched {len(contrib)} query term(s) using TF-IDF.",
            }))
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:top_k]

    # ------------------------------------------------------------------
    # BM25 (Robertson-Sparck Jones)
    # ------------------------------------------------------------------
    def bm25(self, query: str, *, top_k: int = 10,
             language: str | None = None, technology: str | None = None,
             allowed_ids: set[str] | None = None,
             query_weights: dict[str, float] | None = None) -> list[SearchHit]:
        terms = self._terms(query)
        if not terms or self.document_count == 0:
            return []
        allowed = self._allowed(language, technology, allowed_ids)
        qtf = Counter(terms)
        scores: dict[str, float] = {}
        term_contrib: dict[str, dict[str, float]] = {}
        n = self.document_count
        avgdl = self.index.avg_doc_length or 1.0
        for term, qcount in qtf.items():
            postings = self.index.postings.get(term, {})
            df = len(postings)
            if not df:
                continue
            # Positive Robertson-Sparck Jones IDF.
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            q_weight = (query_weights or {}).get(term, 1.0)
            for doc_id, posting in postings.items():
                if allowed is not None and doc_id not in allowed:
                    continue
                dl = self.index.doc_lengths.get(doc_id, avgdl)
                norm = posting.tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                term_score = idf * (posting.tf * (self.k1 + 1.0) / norm)
                term_score *= qcount * q_weight
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score
                term_contrib.setdefault(doc_id, {})[term] = term_score
        hits = []
        for doc_id, score in scores.items():
            if score <= 0:
                continue
            contrib = term_contrib.get(doc_id, {})
            hits.append(self._make_hit(doc_id, score, query, terms, {
                "method": "bm25",
                "matched_terms": list(contrib),
                "term_contributions": {k: round(v, 6) for k, v in contrib.items()},
                "bm25_parameters": {"k1": self.k1, "b": self.b},
                "reason": f"Matched {len(contrib)} query term(s) using BM25 (k1={self.k1}, b={self.b}).",
            }))
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return hits[:top_k]

    # ------------------------------------------------------------------
    # Boolean retrieval
    # ------------------------------------------------------------------
    def boolean(self, query: str, *, top_k: int = 100,
                language: str | None = None, technology: str | None = None) -> list[SearchHit]:
        parsed = parse_query(query)
        postfix = to_postfix(parsed)
        universe = set(self.index.documents)
        stack: list[set[str]] = []
        try:
            for token in postfix:
                if token.kind == "TERM":
                    stack.append(self.index.docs_for_term(token.value))
                elif token.kind == "NOT":
                    if not stack:
                        return []
                    stack.append(universe - stack.pop())
                elif token.kind in {"AND", "OR"}:
                    if len(stack) < 2:
                        return []
                    right, left = stack.pop(), stack.pop()
                    stack.append(left & right if token.kind == "AND" else left | right)
            result_ids = stack[-1] if len(stack) == 1 else set()
        except (IndexError, KeyError):
            return []
        allowed = self._allowed(language, technology, None)
        if allowed is not None:
            result_ids &= allowed
        terms = parsed.terms
        hits = []
        for doc_id in result_ids:
            doc = self.index.documents[doc_id]
            matched = [t for t in terms if doc_id in self.index.postings.get(t, {})]
            # Boolean results are ranked by count of positive matching terms,
            # then source score, for a deterministic and useful order.
            score = float(len(matched))
            hits.append(self._make_hit(doc_id, score, query, terms, {
                "method": "boolean", "matched_terms": matched,
                "reason": "Matched the Boolean expression; terms follow AND/OR/NOT semantics.",
            }))
        hits.sort(key=lambda h: (-h.score,
                                 -(self.index.documents[h.doc_id].get("score") or 0), h.doc_id))
        return hits[:top_k]

    # ------------------------------------------------------------------
    # PRF hook (expanded by SearchPipeline)
    # ------------------------------------------------------------------
    def expand_query(self, query: str, initial_hits: list[SearchHit], *,
                     top_m: int = config.PRF_TOP_M,
                     expansion_terms: int = config.PRF_TERMS) -> tuple[str, list[str]]:
        """Expand a query with frequent discriminative terms from top results.

        Returns (expanded query, newly added terms). The query's original
        terms are protected from duplicate expansion; stopwords and terms that
        occur in most of the corpus are excluded.
        """
        original = self._terms(query)
        seen = set(original)
        candidates: Counter[str] = Counter()
        for hit in initial_hits[:top_m]:
            doc = self.index.documents.get(hit.doc_id)
            if not doc:
                continue
            text = " ".join([doc.get("title", ""), doc.get("error_message", "") or "",
                             doc.get("description", ""), doc.get("solution", "")])
            tf = Counter(canonical_token(t) for t in tokenize(text, remove_stopwords=True))
            for term, count in tf.items():
                if term in seen or len(term) < 3 or not any(c.isalpha() for c in term):
                    continue
                df = len(self.index.postings.get(term, {}))
                if df == 0 or df > self.document_count * 0.35:
                    continue
                candidates[term] += count
        added = [t for t, _ in candidates.most_common(expansion_terms)]
        return (query + (" " + " ".join(added) if added else ""), added)

    # ------------------------------------------------------------------
    # Diagnostics / reference comparison
    # ------------------------------------------------------------------
    def reference_bm25(self, query: str, *, top_k: int = 10,
                       language: str | None = None,
                       technology: str | None = None) -> list[SearchHit]:
        """BM25 reference implementation using a direct document scan.

        This intentionally independent implementation is used in tests to
        compare the posting-based BM25 output (LEX-04).
        """
        terms = Counter(self._terms(query))
        if not terms:
            return []
        allowed = self._allowed(language, technology, None)
        n = self.document_count
        avgdl = self.index.avg_doc_length or 1.0
        results = []
        for doc_id, raw in self.index.documents.items():
            if allowed is not None and doc_id not in allowed:
                continue
            score = 0.0
            for term, qcount in terms.items():
                posting = self.index.postings.get(term, {}).get(doc_id)
                if not posting:
                    continue
                df = len(self.index.postings[term])
                idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                dl = self.index.doc_lengths[doc_id]
                denominator = posting.tf + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += idf * posting.tf * (self.k1 + 1) / denominator * qcount
            if score:
                results.append((doc_id, score))
        results.sort(key=lambda pair: (-pair[1], pair[0]))
        terms_list = list(terms)
        return [self._make_hit(doc_id, score, query, terms_list,
                               {"method": "bm25_reference", "reason": "Reference document-scan BM25."})
                for doc_id, score in results[:top_k]]


def normalize_scores(hits: list[SearchHit]) -> dict[str, float]:
    """Min-max normalize hit scores to [0,1], preserving ties and zeros."""
    if not hits:
        return {}
    values = [h.score for h in hits]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {h.doc_id: (1.0 if hi > 0 else 0.0) for h in hits}
    return {h.doc_id: (h.score - lo) / (hi - lo) for h in hits}
