"""Build deterministic evaluation queries and seed qrels from the real corpus."""
from __future__ import annotations

import argparse
import json

from app import config
from app.evaluation.dataset import (create_evaluation_queries, create_qrels,
                                   create_seed_queries, load_documents, save_qrels,
                                   write_jsonl)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--seeds", type=int, default=config.TARGET_SEEDS)
    args = parser.parse_args()
    config.ensure_dirs()
    docs = load_documents()
    seed_queries = create_seed_queries(docs, args.seeds)
    eval_queries = create_evaluation_queries(docs, min(max(args.queries, 30), 50))
    qrels = create_qrels(eval_queries, docs)
    write_jsonl(config.SEEDS_DIR / "seed_questions.jsonl", seed_queries)
    write_jsonl(config.EVAL_QUERIES_PATH, eval_queries)
    save_qrels(config.QRELS_PATH, qrels)
    provenance = {
        "seed_questions": len(seed_queries),
        "evaluation_queries": len(eval_queries),
        "qrel_rows": len(qrels),
        "qrels_method": "deterministic seed judgments; anchor and shared metadata/error evidence",
        "human_judgments": False,
        "dataset_documents": len(docs),
        "languages": sorted({d.language for d in docs if d.language}),
        "technologies": sorted({t for d in docs for t in d.technologies}),
    }
    with (config.EVAL_DIR / "judgment_provenance.json").open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
