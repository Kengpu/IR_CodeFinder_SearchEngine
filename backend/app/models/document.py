"""CodeFinder — shared document schema (IDX-02).

Every document in the corpus, regardless of source, is normalized into this
schema. The schema is the contract between the crawler, the indexer, the
retrievers, and the frontend.
"""
from __future__ import annotations

import re
from typing import Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

# Traceability: every document must come from a known, allowed source domain.
ALLOWED_SOURCE_DOMAINS = (
    "stackoverflow.com",
    "stackexchange.com",
    "serverfault.com",
    "superuser.com",
    "docs.python.org",
    "numpy.org",
    "pandas.pydata.org",
    "developer.mozilla.org",
    "reactjs.org",
    "react.dev",
    "nodejs.org",
    "github.com",
)

_DOC_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-\.]{2,80}$")


class Document(BaseModel):
    """The shared document schema for the whole pipeline."""

    doc_id: str = Field(..., description="Stable unique id, e.g. so-12345")
    title: str = Field(..., min_length=3, max_length=300)
    error_message: Optional[str] = Field(None, max_length=2000,
                                         description="Verbatim error text/traceback if any")
    description: str = Field(..., min_length=10, max_length=6000,
                             description="Problem description / question body")
    solution: str = Field(..., min_length=10, max_length=8000,
                          description="Accepted or top answer text")
    code_example: Optional[str] = Field(None, max_length=8000,
                                        description="Representative code snippet")
    tags: list[str] = Field(default_factory=list, max_length=10)
    language: Optional[str] = Field(None, max_length=40,
                                    description="Programming language, e.g. python")
    technologies: list[str] = Field(default_factory=list, max_length=10,
                                    description="Frameworks/tools, e.g. react, django")
    source_url: str = Field(..., min_length=10)
    source_site: str = Field(..., min_length=3)
    source_author: Optional[str] = Field(None, max_length=160,
                                        description="Display name of the question author")
    source_author_url: Optional[str] = Field(None, max_length=500,
                                             description="Canonical author profile URL")
    answer_author: Optional[str] = Field(None, max_length=160,
                                         description="Display name of selected solution author")
    answer_author_url: Optional[str] = Field(None, max_length=500,
                                             description="Canonical solution author profile URL")
    created_date: Optional[str] = None
    score: Optional[float] = Field(None, description="Source-site score (votes)")
    license: str = Field(default="CC BY-SA 4.0")
    answer_license: Optional[str] = Field(None, description="License of the selected source answer")

    @field_validator("doc_id")
    @classmethod
    def _valid_doc_id(cls, v: str) -> str:
        if not _DOC_ID_RE.match(v):
            raise ValueError(f"doc_id must match {_DOC_ID_RE.pattern}, got: {v!r}")
        return v

    @field_validator("source_url")
    @classmethod
    def _allowed_domain(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"source_url must be an absolute HTTPS URL: {v!r}")
        host = parsed.hostname.casefold()
        if not any(host == domain or host.endswith("." + domain)
                   for domain in ALLOWED_SOURCE_DOMAINS):
            raise ValueError(f"source_url domain not in allow-list: {v!r}")
        return v

    @field_validator("source_author_url", "answer_author_url")
    @classmethod
    def _safe_profile_url(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        parsed = urlparse(v)
        host = parsed.hostname.casefold() if parsed.hostname else ""
        if parsed.scheme != "https" or not (host == "stackoverflow.com" or host.endswith(".stackoverflow.com")):
            raise ValueError(f"author profile URL must be a Stack Overflow HTTPS URL: {v!r}")
        return v

    def searchable_text(self) -> str:
        """Concatenate all searchable fields in priority order.

        Used for embedding generation and as the default indexing text.
        Title and error_message are repeated/weighted by placement: the
        tokenizer weights early text higher via position-independent TF boost
        applied at indexing time (see indexing/tokenizer.py docstring).
        """
        parts = [self.title]
        if self.error_message:
            parts.append(self.error_message)
        parts.append(self.description)
        parts.append(self.solution)
        if self.code_example:
            parts.append(self.code_example)
        parts.append(" ".join(self.tags))
        parts.append(self.language or "")
        parts.append(" ".join(self.technologies))
        return "\n".join(p for p in parts if p)


class DocumentValidationError(ValueError):
    """Raised when a record cannot be normalized into the schema."""


def validate_documents(docs: list[Document]) -> tuple[list[Document], list[dict]]:
    """Validate a list of records; return (valid, errors).

    Each error dict records doc_id/title/reason so the crawler can log what
    was dropped instead of failing the whole batch (IDX-07).
    """
    valid: list[Document] = []
    errors: list[dict] = []
    seen_ids: set[str] = set()
    for d in docs:
        try:
            if d.doc_id in seen_ids:
                raise DocumentValidationError(f"duplicate doc_id {d.doc_id}")
            seen_ids.add(d.doc_id)
            valid.append(d)
        except Exception as e:  # pydantic raises ValueError subclasses
            errors.append({"doc_id": d.doc_id, "title": d.title[:80], "reason": str(e)})
    return valid, errors
