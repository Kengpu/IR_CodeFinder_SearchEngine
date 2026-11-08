"""Inverted index construction and persistence (IDX-10/11/12).

The index is deliberately transparent rather than delegating to a search
library: each term maps to document postings containing term frequency and
positions. This makes TF, DF, TF-IDF, BM25, Boolean retrieval, and index
inspection reproducible for the project report.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .. import config
from ..models.document import Document
from .tokenizer import canonical_token, tokenize


@dataclass
class Posting:
    """One term/document posting. Positions support future phrase search."""

    tf: int = 0
    positions: list[int] = field(default_factory=list)


@dataclass
class InvertedIndex:
    """Persistable index bundle used by all lexical retrievers."""

    # canonical term -> doc id -> posting
    postings: dict[str, dict[str, Posting]] = field(default_factory=dict)
    doc_lengths: dict[str, int] = field(default_factory=dict)
    documents: dict[str, dict] = field(default_factory=dict)
    metadata: dict[str, dict[str, set[str]]] = field(default_factory=lambda: {
        "language": defaultdict(set), "technology": defaultdict(set), "tag": defaultdict(set)
    })
    total_docs: int = 0
    avg_doc_length: float = 0.0
    vocabulary_size: int = 0

    def __post_init__(self) -> None:
        # defaultdicts are convenient while building but normal dicts pickle
        # and serialize more predictably after loading.
        for key in ("language", "technology", "tag"):
            self.metadata.setdefault(key, {})

    @property
    def document_frequency(self) -> dict[str, int]:
        return {term: len(docs) for term, docs in self.postings.items()}

    def docs_for_term(self, term: str) -> set[str]:
        return set(self.postings.get(canonical_token(term), {}))

    def filtered_doc_ids(self, language: str | None = None,
                         technology: str | None = None,
                         tags: Iterable[str] | None = None) -> set[str] | None:
        """Return the intersection of metadata filters, or None for no filter."""
        sets: list[set[str]] = []
        if language:
            sets.append(set(self.metadata.get("language", {}).get(language.casefold(), set())))
        if technology:
            sets.append(set(self.metadata.get("technology", {}).get(technology.casefold(), set())))
        if tags:
            for tag in tags:
                sets.append(set(self.metadata.get("tag", {}).get(tag.casefold(), set())))
        if not sets:
            return None
        return set.intersection(*sets) if sets else set()

    def save(self, index_path: Path | None = None, doc_lengths_path: Path | None = None,
             meta_path: Path | None = None, sqlite_path: Path | None = None) -> None:
        """Persist binary postings, lengths, metadata JSON, and SQLite rows."""
        index_path = index_path or config.INVERTED_INDEX_PATH
        doc_lengths_path = doc_lengths_path or config.DOC_LENGTHS_PATH
        meta_path = meta_path or config.META_INDEX_PATH
        sqlite_path = sqlite_path or config.SQLITE_PATH
        for p in (index_path, doc_lengths_path, meta_path, sqlite_path):
            p.parent.mkdir(parents=True, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump({"postings": self.postings, "total_docs": self.total_docs,
                         "avg_doc_length": self.avg_doc_length,
                         "vocabulary_size": self.vocabulary_size}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)
        serial_meta = {
            key: {value: sorted(ids) for value, ids in values.items()}
            for key, values in self.metadata.items()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": serial_meta, "documents": self.documents}, f,
                      ensure_ascii=False, indent=2)
        self._save_sqlite(sqlite_path)

    def _save_sqlite(self, path: Path) -> None:
        """SQLite metadata store used for document lookup/detail endpoints."""
        # ``sqlite3.Connection.__exit__`` commits/rolls back but does not close
        # the handle. Explicitly close it so Windows can remove/rebuild the
        # database (and so tests do not leak a file lock).
        con = sqlite3.connect(path)
        try:
            con.executescript("""
                DROP TABLE IF EXISTS documents;
                CREATE TABLE documents (
                    doc_id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    error_message TEXT, description TEXT NOT NULL,
                    solution TEXT NOT NULL, code_example TEXT,
                    tags_json TEXT NOT NULL, language TEXT,
                    technologies_json TEXT NOT NULL, source_url TEXT NOT NULL,
                    source_site TEXT NOT NULL, source_author TEXT,
                    source_author_url TEXT, answer_author TEXT,
                    answer_author_url TEXT, created_date TEXT, score REAL,
                    license TEXT NOT NULL, answer_license TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_documents_language ON documents(language);
            """)
            rows = []
            for raw in self.documents.values():
                rows.append((raw["doc_id"], raw["title"], raw.get("error_message"),
                             raw["description"], raw["solution"], raw.get("code_example"),
                             json.dumps(raw.get("tags", [])), raw.get("language"),
                             json.dumps(raw.get("technologies", [])), raw["source_url"],
                             raw["source_site"], raw.get("source_author"),
                             raw.get("source_author_url"), raw.get("answer_author"),
                             raw.get("answer_author_url"), raw.get("created_date"),
                             raw.get("score"), raw.get("license", "CC BY-SA 4.0"),
                             raw.get("answer_license")))
            con.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            con.commit()
        finally:
            con.close()

    @classmethod
    def load(cls, index_path: Path | None = None, doc_lengths_path: Path | None = None,
             meta_path: Path | None = None) -> "InvertedIndex":
        index_path = index_path or config.INVERTED_INDEX_PATH
        doc_lengths_path = doc_lengths_path or config.DOC_LENGTHS_PATH
        meta_path = meta_path or config.META_INDEX_PATH
        if not index_path.exists() or not doc_lengths_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Lexical index files are missing; run scripts/build_index.py")
        with open(index_path, "rb") as f:
            raw = pickle.load(f)
        with open(doc_lengths_path, "rb") as f:
            lengths = pickle.load(f)
        with open(meta_path, encoding="utf-8") as f:
            meta_raw = json.load(f)
        metadata = {
            key: {value: set(ids) for value, ids in values.items()}
            for key, values in meta_raw.get("metadata", {}).items()
        }
        # Older pickle files may contain plain dict postings; normalize them.
        postings = {}
        for term, docs in raw["postings"].items():
            postings[term] = {}
            for doc_id, posting in docs.items():
                postings[term][doc_id] = posting if isinstance(posting, Posting) else Posting(**posting)
        return cls(postings=postings, doc_lengths=lengths,
                   documents=meta_raw.get("documents", {}), metadata=metadata,
                   total_docs=raw["total_docs"], avg_doc_length=raw["avg_doc_length"],
                   vocabulary_size=raw.get("vocabulary_size", len(postings)))


def load_documents(corpus_path: Path | None = None) -> list[Document]:
    corpus_path = corpus_path or config.CORPUS_PATH
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}; run the crawler first")
    docs = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(Document.model_validate_json(line))
    return docs


def build_index(docs: list[Document], *, remove_stopwords: bool | None = None,
                stemming: bool | None = None) -> InvertedIndex:
    """Build postings, document lengths, and language/technology indexes."""
    idx = InvertedIndex()
    for d in docs:
        raw = d.model_dump(mode="json")
        idx.documents[d.doc_id] = raw
        tokens = tokenize(d.searchable_text(), remove_stopwords=remove_stopwords,
                          stemming=stemming)
        idx.doc_lengths[d.doc_id] = len(tokens)
        # Counter retains TF; second pass stores positions for snippets/phrases.
        counts = Counter(canonical_token(t) for t in tokens)
        positions: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            positions[canonical_token(token)].append(pos)
        for term, tf in counts.items():
            idx.postings.setdefault(term, {})[d.doc_id] = Posting(
                tf=tf, positions=positions[term])
        if d.language:
            idx.metadata.setdefault("language", {}).setdefault(d.language.casefold(), set()).add(d.doc_id)
        for tech in d.technologies:
            idx.metadata.setdefault("technology", {}).setdefault(tech.casefold(), set()).add(d.doc_id)
        for tag in d.tags:
            idx.metadata.setdefault("tag", {}).setdefault(tag.casefold(), set()).add(d.doc_id)

    idx.total_docs = len(docs)
    idx.vocabulary_size = len(idx.postings)
    idx.avg_doc_length = (sum(idx.doc_lengths.values()) / idx.total_docs
                          if idx.total_docs else 0.0)
    return idx


def build_and_save(corpus_path: Path | None = None) -> InvertedIndex:
    config.ensure_dirs()
    docs = load_documents(corpus_path)
    idx = build_index(docs)
    idx.save()
    return idx


if __name__ == "__main__":
    index = build_and_save()
    print(f"indexed {index.total_docs} docs, {index.vocabulary_size} terms")
