"""Generate and persist the FAISS semantic index from the processed corpus."""
from __future__ import annotations

import argparse

from app import config
from app.indexing.inverted import load_documents
from app.retrieval.semantic import SemanticSearcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config.DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()
    config.ensure_dirs()
    docs = load_documents()
    if not docs:
        raise SystemExit("Corpus has no documents. Collect and process data first.")
    searcher = SemanticSearcher.from_documents(docs, model_name=args.model)
    searcher.save()
    print(f"Saved {len(searcher.doc_ids)} normalized vectors with {args.model}")


if __name__ == "__main__":
    main()
