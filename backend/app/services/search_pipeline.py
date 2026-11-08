"""Integrated retrieval and ranking pipeline (RNK-10).

Search mode names are the public API contract. This class coordinates the
independent lexical and semantic retrievers, applies filters consistently,
performs candidate fusion/re-ranking, and attaches explanations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import config
from ..indexing.inverted import InvertedIndex
from ..indexing.tokenizer import canonical_token, tokenize
from ..ranking.fusion import (field_rerank, reciprocal_rank_fusion,
                              weighted_score_fusion)
from ..retrieval.lexical import LexicalSearcher, SearchHit
from ..retrieval.semantic import SemanticSearcher
from ..utils.text import make_snippet

SEARCH_MODES = {
    "tfidf", "bm25", "boolean", "semantic", "hybrid_rrf",
    "hybrid_weighted", "hybrid_prf",
}


@dataclass
class PipelineResult:
    hits: list[SearchHit]
    mode: str
    query: str
    metadata: dict[str, Any]


class SearchPipeline:
    def __init__(self, lexical: LexicalSearcher,
                 semantic: SemanticSearcher | None = None):
        self.lexical = lexical
        self.semantic = semantic
        self.documents = lexical.index.documents

    def search(self, query: str, *, mode: str = "hybrid_rrf", top_k: int = 10,
               language: str | None = None, technology: str | None = None,
               prf: bool | None = None, rerank: bool = True,
               lexical_weight: float = config.LEXICAL_WEIGHT,
               semantic_weight: float = config.SEMANTIC_WEIGHT) -> PipelineResult:
        if mode not in SEARCH_MODES:
            raise ValueError(f"Unsupported mode '{mode}'. Choose from {sorted(SEARCH_MODES)}")
        if not query or not query.strip():
            return PipelineResult([], mode, query or "", {"warning": "empty_query"})
        if top_k < 1 or top_k > 100:
            raise ValueError("top_k must be between 1 and 100")

        metadata: dict[str, Any] = {"mode": mode}
        lexical_hits: list[SearchHit] = []
        semantic_hits: list[SearchHit] = []
        fused_hits: list[SearchHit]
        candidate_k = max(top_k * 5, 50)

        if mode == "tfidf":
            fused_hits = self.lexical.tfidf(query, top_k=candidate_k,
                                            language=language, technology=technology)
        elif mode == "bm25":
            fused_hits = self.lexical.bm25(query, top_k=candidate_k,
                                           language=language, technology=technology)
        elif mode == "boolean":
            fused_hits = self.lexical.boolean(query, top_k=candidate_k,
                                              language=language, technology=technology)
        elif mode == "semantic":
            self._require_semantic()
            fused_hits = self.semantic.search(query, top_k=candidate_k,
                                              language=language, technology=technology)
        else:
            self._require_semantic()
            lexical_hits = self.lexical.bm25(query, top_k=candidate_k,
                                             language=language, technology=technology)
            semantic_hits = self.semantic.search(query, top_k=candidate_k,
                                                 language=language, technology=technology)
            metadata.update({"lexical_candidates": len(lexical_hits),
                             "semantic_candidates": len(semantic_hits)})
            if mode == "hybrid_rrf":
                fused_hits = reciprocal_rank_fusion(
                    [lexical_hits, semantic_hits], k=config.RRF_K,
                    window=config.RRF_WINDOW, top_k=candidate_k)
            elif mode == "hybrid_weighted":
                fused_hits = weighted_score_fusion(
                    lexical_hits, semantic_hits, lexical_weight=lexical_weight,
                    semantic_weight=semantic_weight, top_k=candidate_k)
            else:  # hybrid_prf: feedback is explicit or enabled by default.
                use_prf = config.PRF_ENABLED if prf is None else prf
                if use_prf:
                    expansion_source = lexical_hits or semantic_hits
                    expanded, added = self.lexical.expand_query(
                        query, expansion_source, top_m=config.PRF_TOP_M,
                        expansion_terms=config.PRF_TERMS)
                    metadata.update({"prf_enabled": True, "expanded_query": expanded,
                                    "expansion_terms": added})
                    expanded_lexical = self.lexical.bm25(
                        expanded, top_k=candidate_k, language=language,
                        technology=technology)
                    fused_hits = reciprocal_rank_fusion(
                        [expanded_lexical, semantic_hits], k=config.RRF_K,
                        window=config.RRF_WINDOW, top_k=candidate_k)
                    lexical_hits = expanded_lexical
                else:
                    metadata["prf_enabled"] = False
                    fused_hits = reciprocal_rank_fusion(
                        [lexical_hits, semantic_hits], k=config.RRF_K,
                        window=config.RRF_WINDOW, top_k=candidate_k)

        if rerank and mode not in {"boolean"} and fused_hits:
            terms = [canonical_token(t) for t in tokenize(query, remove_stopwords=False)
                     if t.upper() not in {"AND", "OR", "NOT"} and t not in {"(", ")"}]
            fused_hits = field_rerank(fused_hits, self.documents, terms)
        fused_hits = fused_hits[:top_k]
        metadata["result_count"] = len(fused_hits)
        return PipelineResult(fused_hits, mode, query, metadata)

    def similar(self, doc_id: str, *, top_k: int = 5,
                language: str | None = None,
                technology: str | None = None) -> list[SearchHit]:
        if self.semantic:
            return self.semantic.similar(doc_id, top_k=top_k,
                                         language=language, technology=technology)
        # Dense artifacts may not yet have been built. Provide a real lexical
        # related-document fallback and expose its method to the API caller.
        raw = self.documents.get(doc_id)
        if not raw:
            return []
        query = " ".join([raw.get("title", ""), raw.get("error_message") or ""])
        hits = self.lexical.bm25(query, top_k=top_k + 1,
                                 language=language, technology=technology)
        return [h for h in hits if h.doc_id != doc_id][:top_k]

    def _require_semantic(self) -> None:
        if self.semantic is None:
            raise RuntimeError("Semantic index is unavailable. Build it with scripts/build_embeddings.py.")


def build_pipeline(*, require_semantic: bool = False) -> SearchPipeline:
    """Load persisted indexes into an application-ready search pipeline."""
    index = InvertedIndex.load()
    lexical = LexicalSearcher(index)
    semantic = None
    try:
        semantic = SemanticSearcher.load(documents=index.documents)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if require_semantic:
            raise
        print(f"[search] semantic index unavailable: {exc}")
    return SearchPipeline(lexical, semantic)


__all__ = ["SearchPipeline", "PipelineResult", "SEARCH_MODES", "build_pipeline"]
