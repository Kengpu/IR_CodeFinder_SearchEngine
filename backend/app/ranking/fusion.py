"""Candidate fusion and ranking utilities (RNK-04..09).

The functions here deliberately accept ranked ``SearchHit`` lists rather than
coupling fusion to a specific retriever. This keeps RRF and weighted fusion
reproducible and makes ablation experiments straightforward.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..retrieval.lexical import SearchHit, normalize_scores


def deduplicate_hits(*hit_lists: Iterable[SearchHit]) -> dict[str, SearchHit]:
    """Keep one hit per document, retaining the strongest source score."""
    unique: dict[str, SearchHit] = {}
    for hits in hit_lists:
        for hit in hits:
            old = unique.get(hit.doc_id)
            if old is None or hit.score > old.score:
                unique[hit.doc_id] = hit
    return unique


def reciprocal_rank_fusion(result_lists: list[list[SearchHit]], *,
                           k: int = 60, window: int = 100,
                           top_k: int = 10) -> list[SearchHit]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    RRF(doc) = sum(1 / (k + rank_i(doc))). Ranks are one-based and each
    input list is truncated at ``window`` to avoid very deep candidates
    overwhelming high-confidence results.
    """
    scores: defaultdict[str, float] = defaultdict(float)
    source_ranks: defaultdict[str, dict[str, int]] = defaultdict(dict)
    source_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
    for source_no, hits in enumerate(result_lists):
        for rank, hit in enumerate(hits[:window], start=1):
            scores[hit.doc_id] += 1.0 / (k + rank)
            source_ranks[hit.doc_id][str(source_no)] = rank
            source_scores[hit.doc_id][str(source_no)] = hit.score
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    hits: list[SearchHit] = []
    for doc_id in ordered:
        hits.append(SearchHit(
            doc_id=doc_id,
            score=scores[doc_id],
            explanation={
                "method": "rrf",
                "rrf_score": scores[doc_id],
                "source_ranks": dict(source_ranks[doc_id]),
                "source_scores": dict(source_scores[doc_id]),
                "reason": "Combined lexical and semantic rank positions with reciprocal rank fusion.",
            },
        ))
    return hits


def weighted_score_fusion(lexical_hits: list[SearchHit],
                          semantic_hits: list[SearchHit], *,
                          lexical_weight: float = 0.55,
                          semantic_weight: float = 0.45,
                          top_k: int = 10) -> list[SearchHit]:
    """Fuse min-max-normalized lexical and semantic scores."""
    if lexical_weight < 0 or semantic_weight < 0 or lexical_weight + semantic_weight <= 0:
        raise ValueError("Fusion weights must be non-negative and not both zero")
    total = lexical_weight + semantic_weight
    lexical_weight /= total
    semantic_weight /= total
    lexical_norm = normalize_scores(lexical_hits)
    semantic_norm = normalize_scores(semantic_hits)
    all_ids = set(lexical_norm) | set(semantic_norm)
    final: dict[str, float] = {}
    for doc_id in all_ids:
        final[doc_id] = (lexical_weight * lexical_norm.get(doc_id, 0.0)
                         + semantic_weight * semantic_norm.get(doc_id, 0.0))
    ordered = sorted(final, key=lambda doc_id: (-final[doc_id], doc_id))[:top_k]
    lex_raw = {h.doc_id: h.score for h in lexical_hits}
    sem_raw = {h.doc_id: h.score for h in semantic_hits}
    return [SearchHit(
        doc_id=doc_id,
        score=final[doc_id],
        explanation={
            "method": "weighted",
            "lexical_score": lex_raw.get(doc_id, 0.0),
            "semantic_score": sem_raw.get(doc_id, 0.0),
            "normalized_lexical_score": lexical_norm.get(doc_id, 0.0),
            "normalized_semantic_score": semantic_norm.get(doc_id, 0.0),
            "lexical_weight": lexical_weight,
            "semantic_weight": semantic_weight,
            "reason": "Min-max-normalized lexical and semantic scores were combined by weighted sum.",
        },
    ) for doc_id in ordered]


def field_rerank(hits: list[SearchHit], documents: dict[str, dict], query_terms: list[str],
                 *, weight: float = 0.30) -> list[SearchHit]:
    """Apply a transparent, lightweight field-aware re-ranking pass.

    Error messages and titles are more diagnostic than long answer text. The
    re-ranker adds a bounded overlap feature and records every component in
    the explanation, rather than hiding a heuristic behind an opaque score.
    """
    if not hits:
        return []
    q = {t.casefold() for t in query_terms if t}
    raw_scores = [h.score for h in hits]
    lo, hi = min(raw_scores), max(raw_scores)
    span = hi - lo
    output: list[SearchHit] = []
    for hit in hits:
        doc = documents.get(hit.doc_id, {})
        title = (doc.get("title") or "").casefold()
        error = (doc.get("error_message") or "").casefold()
        title_overlap = sum(1 for t in q if t in title) / max(len(q), 1)
        error_overlap = sum(1 for t in q if t in error) / max(len(q), 1)
        field_score = min(1.0, 0.65 * title_overlap + 0.35 * error_overlap)
        base = ((hit.score - lo) / span) if span else (1.0 if hit.score > 0 else 0.0)
        final = (1.0 - weight) * base + weight * field_score
        explanation = dict(hit.explanation)
        explanation.update({
            "base_score": hit.score,
            "reranker_score": field_score,
            "final_score": final,
            "title_term_overlap": title_overlap,
            "error_term_overlap": error_overlap,
            "reason": explanation.get("reason", "") +
                      " A bounded title/error-field overlap re-ranker was applied.",
        })
        output.append(SearchHit(hit.doc_id, final, explanation, hit.snippet))
    output.sort(key=lambda h: (-h.score, h.doc_id))
    return output


def rank_explanation(hit: SearchHit) -> dict:
    """Return a stable JSON-friendly explanation payload."""
    explanation = dict(hit.explanation)
    explanation.setdefault("final_score", hit.score)
    explanation.setdefault("reason", "Ranked by the selected retrieval method.")
    return explanation


__all__ = ["deduplicate_hits", "reciprocal_rank_fusion", "weighted_score_fusion",
           "field_rerank", "rank_explanation"]
