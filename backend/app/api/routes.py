"""Public HTTP endpoints and response mapping."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from .. import config
from ..indexing.tokenizer import tokenize
from ..models.api import (DocumentResponse, EvaluationResponse, HealthResponse,
                          SearchResponse, SearchResponseDocument, SimilarResponse)
from ..ranking.fusion import rank_explanation
from ..retrieval.lexical import SearchHit
from ..utils.text import highlight_text
from ..services.search_pipeline import SEARCH_MODES, SearchPipeline

router = APIRouter()


def _pipeline(request: Request) -> SearchPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        detail = getattr(request.app.state, "load_error", None) or "Search indexes are not available."
        raise HTTPException(status_code=503, detail=f"{detail} Build indexes before starting the API.")
    return pipeline


def _serialize_hit(pipeline: SearchPipeline, hit: SearchHit, query: str) -> SearchResponseDocument:
    raw = pipeline.documents.get(hit.doc_id)
    if raw is None:
        raise HTTPException(status_code=500, detail=f"Index returned unknown document id {hit.doc_id}")
    terms = [t for t in tokenize(query, remove_stopwords=False)
             if t.upper() not in {"AND", "OR", "NOT"} and t not in {"(", ")"}]
    snippet = hit.snippet or raw.get("error_message") or raw.get("description", "")[:240]
    return SearchResponseDocument(
        doc_id=hit.doc_id,
        title=raw["title"],
        error_message=raw.get("error_message"),
        description=raw["description"],
        solution=raw["solution"],
        code_example=raw.get("code_example"),
        tags=raw.get("tags", []),
        language=raw.get("language"),
        technologies=raw.get("technologies", []),
        source_url=raw["source_url"],
        source_site=raw.get("source_site", "unknown"),
        source_author=raw.get("source_author"),
        source_author_url=raw.get("source_author_url"),
        answer_author=raw.get("answer_author"),
        answer_author_url=raw.get("answer_author_url"),
        license=raw.get("license", "CC BY-SA 4.0"),
        answer_license=raw.get("answer_license"),
        score=float(hit.score),
        ranking_explanation=rank_explanation(hit),
        snippet=snippet,
        highlighted_snippet=highlight_text(snippet, terms),
    )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    pipeline = getattr(request.app.state, "pipeline", None)
    lexical_ok = pipeline is not None
    semantic_ok = bool(pipeline and pipeline.semantic)
    return HealthResponse(
        status="ok" if lexical_ok else "degraded",
        lexical_index=lexical_ok,
        semantic_index=semantic_ok,
        documents=len(pipeline.documents) if pipeline else 0,
    )


@router.get("/search", response_model=SearchResponse, tags=["search"])
def search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=2000, description="Error text or developer question"),
    mode: str = Query("hybrid_rrf", description="One of: tfidf, bm25, boolean, semantic, hybrid_rrf, hybrid_weighted, hybrid_prf"),
    top_k: int = Query(10, ge=1, le=100),
    language: str | None = Query(None, max_length=40),
    technology: str | None = Query(None, max_length=80),
    prf: bool | None = None,
    rerank: bool = True,
    lexical_weight: float = Query(0.55, ge=0, le=1),
    semantic_weight: float = Query(0.45, ge=0, le=1),
) -> SearchResponse:
    q = q.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Query cannot be blank")
    if mode not in SEARCH_MODES:
        raise HTTPException(status_code=422, detail=f"Unsupported mode. Use one of {sorted(SEARCH_MODES)}")
    pipeline = _pipeline(request)
    try:
        result = pipeline.search(
            q, mode=mode, top_k=top_k, language=language, technology=technology,
            prf=prf, rerank=rerank,
            lexical_weight=lexical_weight, semantic_weight=semantic_weight,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    results = [_serialize_hit(pipeline, hit, q) for hit in result.hits]
    return SearchResponse(query=q, mode=result.mode, results=results,
                          total=len(results), metadata=result.metadata)


@router.get("/documents/{document_id}", response_model=DocumentResponse, tags=["documents"])
def get_document(request: Request, document_id: str) -> DocumentResponse:
    pipeline = _pipeline(request)
    raw = pipeline.documents.get(document_id)
    if raw is None:
        raise HTTPException(status_code=404, detail="Document not found")
    base_hit = SearchHit(document_id, 0.0, {
        "method": "document_detail", "reason": "Full document details.", "final_score": 0.0,
    }, raw.get("error_message") or raw.get("description", "")[:240])
    base = _serialize_hit(pipeline, base_hit, "")
    related = pipeline.similar(document_id, top_k=5)
    related_docs = [_serialize_hit(pipeline, hit, raw.get("title", "")) for hit in related]
    return DocumentResponse(**base.model_dump(), related_errors=related_docs)


@router.get("/similar/{document_id}", response_model=SimilarResponse, tags=["documents"])
def similar(request: Request, document_id: str,
            top_k: int = Query(5, ge=1, le=20),
            language: str | None = Query(None, max_length=40),
            technology: str | None = Query(None, max_length=80)) -> SimilarResponse:
    pipeline = _pipeline(request)
    if document_id not in pipeline.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    hits = pipeline.similar(document_id, top_k=top_k, language=language, technology=technology)
    return SimilarResponse(document_id=document_id,
                           results=[_serialize_hit(pipeline, hit, "") for hit in hits],
                           total=len(hits))


@router.get("/technologies", tags=["metadata"])
def technologies(request: Request) -> list[dict[str, Any]]:
    pipeline = _pipeline(request)
    counts: dict[str, int] = {}
    for raw in pipeline.documents.values():
        for name in raw.get("technologies", []):
            counts[name] = counts.get(name, 0) + 1
    return [{"name": k, "count": v} for k, v in sorted(counts.items(), key=lambda p: (-p[1], p[0]))]


@router.get("/languages", tags=["metadata"])
def languages(request: Request) -> list[dict[str, Any]]:
    pipeline = _pipeline(request)
    counts: dict[str, int] = {}
    for raw in pipeline.documents.values():
        name = raw.get("language")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return [{"name": k, "count": v} for k, v in sorted(counts.items(), key=lambda p: (-p[1], p[0]))]


@router.get("/evaluation", response_model=EvaluationResponse, tags=["evaluation"])
def evaluation() -> EvaluationResponse:
    """Return saved evaluation results only; never synthesize metrics."""
    candidates = [config.EVAL_RESULTS_DIR / "summary.json",
                  config.EVAL_DIR / "summary.json"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return EvaluationResponse(available=False, metrics={},
                                  message="Evaluation has not been run yet. Use scripts/evaluate.py.")
    try:
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return EvaluationResponse(available=False, metrics={},
                                  message=f"Could not read evaluation results: {exc}")
    return EvaluationResponse(available=True, metrics=metrics)


__all__ = ["router"]
