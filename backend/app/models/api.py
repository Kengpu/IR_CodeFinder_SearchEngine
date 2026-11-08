"""Pydantic request/response models for the public FastAPI contract."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


SearchMode = Literal[
    "tfidf", "bm25", "boolean", "semantic", "hybrid_rrf",
    "hybrid_weighted", "hybrid_prf",
]


class SearchResponseDocument(BaseModel):
    doc_id: str
    title: str
    error_message: str | None = None
    description: str
    solution: str
    code_example: str | None = None
    tags: list[str] = Field(default_factory=list)
    language: str | None = None
    technologies: list[str] = Field(default_factory=list)
    source_url: str
    source_site: str
    source_author: str | None = None
    source_author_url: str | None = None
    answer_author: str | None = None
    answer_author_url: str | None = None
    license: str = "CC BY-SA 4.0"
    answer_license: str | None = None
    score: float
    ranking_explanation: dict[str, Any] = Field(default_factory=dict)
    snippet: str = ""
    highlighted_snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResponseDocument]
    total: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    lexical_index: bool
    semantic_index: bool
    documents: int
    version: str = "1.0.0"


class TechnologyCount(BaseModel):
    name: str
    count: int


class EvaluationResponse(BaseModel):
    available: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class SearchQuery(BaseModel):
    q: str = Field(..., min_length=1, max_length=2000)
    mode: SearchMode = "hybrid_rrf"
    top_k: int = Field(10, ge=1, le=100)
    language: str | None = Field(None, max_length=40)
    technology: str | None = Field(None, max_length=80)
    prf: bool | None = None
    rerank: bool = True
    lexical_weight: float = Field(0.55, ge=0, le=1)
    semantic_weight: float = Field(0.45, ge=0, le=1)

    @field_validator("q")
    @classmethod
    def non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("q cannot be blank")
        return value


class SimilarResponse(BaseModel):
    document_id: str
    results: list[SearchResponseDocument]
    total: int


class DocumentResponse(SearchResponseDocument):
    related_errors: list[SearchResponseDocument] = Field(default_factory=list)


__all__ = ["SearchMode", "SearchResponseDocument", "SearchResponse", "ErrorResponse",
           "HealthResponse", "TechnologyCount", "EvaluationResponse", "SearchQuery",
           "SimilarResponse", "DocumentResponse"]
