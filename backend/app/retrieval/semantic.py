"""Dense semantic retrieval with Sentence Transformers + FAISS (DEN-01..12).

Documents and query vectors are L2-normalized; FAISS inner product therefore
returns cosine similarity. The persisted ID list is the authoritative mapping
from FAISS row number to document id, preventing embedding/id misalignment.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..indexing.inverted import InvertedIndex
from ..models.document import Document
from ..utils.text import make_snippet
from .lexical import SearchHit


class SemanticSearcher:
    """FAISS-backed cosine searcher; model is injectable for deterministic tests."""

    def __init__(self, index: Any, doc_ids: list[str], documents: dict[str, dict],
                 model: Any, *, model_name: str = config.DEFAULT_EMBEDDING_MODEL):
        if len(doc_ids) != int(index.ntotal):
            raise ValueError(f"FAISS index has {index.ntotal} vectors but {len(doc_ids)} ids")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids contains duplicates; embeddings must map 1:1")
        missing = set(doc_ids) - set(documents)
        if missing:
            raise ValueError(f"Embedding ids missing document metadata: {sorted(missing)[:3]}")
        self.index = index
        self.doc_ids = doc_ids
        self.documents = documents
        self.model = model
        self.model_name = model_name
        self.row_by_id = {doc_id: row for row, doc_id in enumerate(doc_ids)}

    @classmethod
    def from_documents(cls, docs: list[Document], model_name: str | None = None,
                       batch_size: int = config.EMBEDDING_BATCH_SIZE) -> "SemanticSearcher":
        """Encode documents and create an in-memory FAISS cosine index."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Semantic search requires sentence-transformers and faiss-cpu; "
                               "install backend/requirements.txt") from exc
        model_name = model_name or config.DEFAULT_EMBEDDING_MODEL
        model = SentenceTransformer(model_name)
        texts = [d.searchable_text() for d in docs]
        vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, [d.doc_id for d in docs],
                   {d.doc_id: d.model_dump(mode="json") for d in docs}, model,
                   model_name=model_name)

    @classmethod
    def load(cls, documents: dict[str, dict] | None = None,
             *, faiss_path: Path | None = None,
             ids_path: Path | None = None) -> "SemanticSearcher":
        """Load persisted FAISS index and row-aligned ID manifest."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Semantic search requires sentence-transformers and faiss-cpu") from exc
        faiss_path = faiss_path or config.FAISS_INDEX_PATH
        ids_path = ids_path or config.DOC_IDS_PATH
        if not faiss_path.exists() or not ids_path.exists():
            raise FileNotFoundError("Semantic index files are missing; run scripts/build_embeddings.py")
        with open(ids_path, encoding="utf-8") as f:
            manifest = json.load(f)
        doc_ids = manifest["doc_ids"]
        model_name = manifest.get("model_name", config.DEFAULT_EMBEDDING_MODEL)
        if documents is None:
            lexical_meta = config.META_INDEX_PATH
            if not lexical_meta.exists():
                raise FileNotFoundError("Document metadata missing; build lexical index first")
            with open(lexical_meta, encoding="utf-8") as f:
                documents = json.load(f).get("documents", {})
        faiss_index = faiss.read_index(str(faiss_path))
        model = SentenceTransformer(model_name)
        return cls(faiss_index, doc_ids, documents, model, model_name=model_name)

    def save(self, *, faiss_path: Path | None = None, ids_path: Path | None = None,
             vectors_path: Path | None = None) -> None:
        """Persist FAISS index and aligned vector/id artifacts."""
        import faiss
        faiss_path = faiss_path or config.FAISS_INDEX_PATH
        ids_path = ids_path or config.DOC_IDS_PATH
        vectors_path = vectors_path or config.DOC_EMBEDDINGS_PATH
        for p in (faiss_path, ids_path, vectors_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(faiss_path))
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump({"doc_ids": self.doc_ids, "model_name": self.model_name}, f,
                      ensure_ascii=False, indent=2)
        vectors = np.vstack([self.index.reconstruct(i) for i in range(self.index.ntotal)])
        np.save(vectors_path, vectors)
        with open(config.EMBEDDING_MODEL_PATH, "w", encoding="utf-8") as f:
            f.write(self.model_name)

    def _encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], convert_to_numpy=True,
                                   normalize_embeddings=True)
        return np.ascontiguousarray(vector, dtype="float32")

    def search(self, query: str, *, top_k: int = 10,
               language: str | None = None, technology: str | None = None,
               exclude_ids: set[str] | None = None) -> list[SearchHit]:
        """Return cosine-ranked results; filtered docs are skipped after FAISS."""
        if not query.strip() or self.index.ntotal == 0 or top_k <= 0:
            return []
        from ..indexing.tokenizer import canonical_token, tokenize
        terms = [t for t in tokenize(query, remove_stopwords=False)
                 if t.upper() not in {"AND", "OR", "NOT"}]
        q = self._encode_query(query)
        # Over-fetch so language/technology filters still have enough results.
        requested = self.index.ntotal if language or technology else min(
            self.index.ntotal,
            max(top_k * config.FAISS_CANDIDATES_MULTIPLIER, top_k),
        )
        scores, rows = self.index.search(q, requested)
        output: list[SearchHit] = []
        for score, row in zip(scores[0].tolist(), rows[0].tolist()):
            if row < 0:
                continue
            doc_id = self.doc_ids[row]
            if exclude_ids and doc_id in exclude_ids:
                continue
            raw = self.documents[doc_id]
            if language and (raw.get("language") or "").casefold() != language.casefold():
                continue
            if technology and technology.casefold() not in {
                    t.casefold() for t in raw.get("technologies", [])}:
                continue
            source = raw.get("error_message") or raw.get("description", "")
            snippet = make_snippet(source, terms)
            output.append(SearchHit(doc_id, float(score), {
                "method": "semantic", "semantic_score": round(float(score), 6),
                "embedding_model": self.model_name,
                "reason": f"Semantically similar query/document embeddings (cosine={score:.3f}).",
            }, snippet))
            if len(output) >= top_k:
                break
        return output

    def similar(self, doc_id: str, *, top_k: int = 5,
                language: str | None = None,
                technology: str | None = None) -> list[SearchHit]:
        """Find semantically similar documents, excluding the source document."""
        if doc_id not in self.row_by_id:
            return []
        raw = self.documents[doc_id]
        query = " ".join([raw.get("title", ""), raw.get("error_message") or "",
                           raw.get("description", "")])
        vector = self.index.reconstruct(self.row_by_id[doc_id]).reshape(1, -1).astype("float32")
        # Vector is already normalized when built; normalize defensively.
        norm = np.linalg.norm(vector)
        if norm:
            vector /= norm
        scores, rows = self.index.search(vector, min(self.index.ntotal, top_k + 1))
        out = []
        for score, row in zip(scores[0].tolist(), rows[0].tolist()):
            if row < 0:
                continue
            candidate_id = self.doc_ids[row]
            if candidate_id == doc_id:
                continue
            candidate = self.documents[candidate_id]
            if language and (candidate.get("language") or "").casefold() != language.casefold():
                continue
            if technology and technology.casefold() not in {
                    t.casefold() for t in candidate.get("technologies", [])}:
                continue
            out.append(SearchHit(candidate_id, float(score), {
                "method": "similar", "semantic_score": round(float(score), 6),
                "reason": f"Similar to {doc_id} by embedding cosine similarity ({score:.3f}).",
            }, candidate.get("error_message") or candidate.get("description", "")[:240]))
            if len(out) >= top_k:
                break
        return out


def compare_models(docs: list[Document], queries: list[str],
                   model_names: list[str]) -> dict[str, dict[str, float]]:
    """Compare embedding models using mean pairwise similarity consistency.

    For each model, encode the same document/query pairs and report query-to-
    top-document cosine plus encoding time. Retrieval quality comparisons use
    the shared qrels evaluator (see scripts/compare_models.py); this lightweight
    helper reports reproducible model size/latency and similarity behavior.
    """
    import time
    from sentence_transformers import SentenceTransformer
    reports = {}
    doc_texts = [d.searchable_text() for d in docs]
    for name in model_names:
        started = time.perf_counter()
        model = SentenceTransformer(name)
        doc_vec = model.encode(doc_texts, convert_to_numpy=True, normalize_embeddings=True,
                               show_progress_bar=False)
        query_vec = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)
        sims = query_vec @ doc_vec.T
        reports[name] = {
            "documents": float(len(docs)), "queries": float(len(queries)),
            "encoding_seconds": time.perf_counter() - started,
            "mean_best_cosine": float(sims.max(axis=1).mean()) if len(queries) else 0.0,
            "embedding_dimension": float(doc_vec.shape[1]) if len(docs) else 0.0,
        }
    return reports


def save_embeddings(searcher: SemanticSearcher) -> None:
    searcher.save()


__all__ = ["SemanticSearcher", "compare_models", "save_embeddings"]
