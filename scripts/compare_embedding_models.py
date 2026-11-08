"""Compare two real embedding models using the same corpus, queries, and qrels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app import config
from app.evaluation.dataset import load_qrels, read_jsonl
from app.evaluation.metrics import evaluate_query_set
from app.indexing.inverted import load_documents


def evaluate_model(model_name: str, docs, queries, qrels) -> dict:
    started = time.perf_counter()
    model = SentenceTransformer(model_name)
    doc_vectors = model.encode([d.searchable_text() for d in docs],
                               batch_size=config.EMBEDDING_BATCH_SIZE,
                               show_progress_bar=True, convert_to_numpy=True,
                               normalize_embeddings=True)
    query_vectors = model.encode([q["query"] for q in queries],
                                 batch_size=config.EMBEDDING_BATCH_SIZE,
                                 show_progress_bar=False, convert_to_numpy=True,
                                 normalize_embeddings=True)
    vectors = np.ascontiguousarray(doc_vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    _, rows = index.search(np.ascontiguousarray(query_vectors, dtype="float32"),
                           min(100, len(docs)))
    rankings = {q["query_id"]: [docs[int(row)].doc_id for row in row_list if row >= 0]
                for q, row_list in zip(queries, rows)}
    metrics = evaluate_query_set(rankings, qrels, (5, 10))
    metrics.update({"embedding_dimension": int(vectors.shape[1]),
                    "encoding_and_search_seconds": time.perf_counter() - started,
                    "model": model_name})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs=2, default=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
    ])
    parser.add_argument("--output", type=Path, default=config.EVAL_RESULTS_DIR / "embedding_model_comparison.json")
    args = parser.parse_args()
    docs = load_documents()
    queries = read_jsonl(config.EVAL_QUERIES_PATH)
    qrels = load_qrels(config.QRELS_PATH)
    report = {name: evaluate_model(name, docs, queries, qrels) for name in args.models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
