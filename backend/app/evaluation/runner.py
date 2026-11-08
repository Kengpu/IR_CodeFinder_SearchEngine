"""Run reproducible retrieval comparisons and save measured metrics."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config
from ..indexing.inverted import InvertedIndex
from ..retrieval.lexical import LexicalSearcher
from ..retrieval.semantic import SemanticSearcher
from ..services.search_pipeline import SearchPipeline
from .dataset import load_qrels, read_jsonl
from .graph import graph_report
from .metrics import evaluate_query_set, percentile


def _make_pipeline() -> SearchPipeline:
    index = InvertedIndex.load()
    lexical = LexicalSearcher(index)
    semantic = SemanticSearcher.load(documents=index.documents)
    return SearchPipeline(lexical, semantic)


def run_evaluation(*, output_path: Path | None = None,
                   include_modes: tuple[str, ...] = ("tfidf", "bm25", "semantic",
                                                      "hybrid_rrf", "hybrid_weighted",
                                                      "hybrid_prf")) -> dict:
    queries = read_jsonl(config.EVAL_QUERIES_PATH)
    qrels = load_qrels(config.QRELS_PATH)
    if not queries or not qrels:
        raise FileNotFoundError("Evaluation queries/qrels missing; run scripts/prepare_evaluation.py")
    pipeline = _make_pipeline()
    mode_reports: dict[str, dict] = {}
    mode_rankings: dict[str, dict[str, list[str]]] = {}
    for mode in include_modes:
        rankings: dict[str, list[str]] = {}
        latencies: list[float] = []
        for item in queries:
            started = time.perf_counter()
            result = pipeline.search(item["query"], mode=mode, top_k=100, rerank=True)
            latencies.append((time.perf_counter() - started) * 1000)
            rankings[item["query_id"]] = [hit.doc_id for hit in result.hits]
        mode_rankings[mode] = rankings
        metrics = evaluate_query_set(rankings, qrels, k_values=(5, 10))
        metrics["latency_mean_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
        metrics["latency_p95_ms"] = percentile(latencies, 95)
        mode_reports[mode] = metrics
    ablations = {}
    # PRF is measured against the exact same query/qrel set and ranking code.
    rankings: dict[str, list[str]] = {}
    latencies = []
    for item in queries:
        started = time.perf_counter()
        result = pipeline.search(item["query"], mode="hybrid_prf", top_k=100,
                                 prf=False)
        latencies.append((time.perf_counter() - started) * 1000)
        rankings[item["query_id"]] = [hit.doc_id for hit in result.hits]
    ablations["hybrid_prf_disabled"] = evaluate_query_set(rankings, qrels, (5, 10))
    ablations["hybrid_prf_disabled"]["latency_mean_ms"] = (
        sum(latencies) / len(latencies) if latencies else 0.0)
    if "hybrid_prf" in mode_reports:
        ablations["hybrid_prf_enabled"] = mode_reports["hybrid_prf"]
    report = {
        "experiment": "CodeFinder retrieval evaluation",
        "dataset_documents": len(pipeline.documents),
        "evaluation_query_count": len(queries),
        "qrel_query_count": len(qrels),
        "judgments": "deterministic seed qrels; not expert human judgments",
        "judgment_provenance": "data/evaluation/judgment_provenance.json",
        "embedding_model": pipeline.semantic.model_name if pipeline.semantic else None,
        "metrics": mode_reports,
        "ablation": ablations,
        "limitations": [
            "Seed qrels are deterministic corpus-derived judgments and must not be interpreted as human relevance assessments.",
            "Evaluation queries include anchor-derived text; this can favor lexical matching and creates a known selection bias.",
            "For formal claims, independently judge a held-out query set with multiple assessors.",
        ],
    }
    output_path = output_path or config.EVAL_RESULTS_DIR / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    # Persist complete run details for reproducibility and plotting.
    with (output_path.parent / "rankings.json").open("w", encoding="utf-8") as stream:
        json.dump(mode_rankings, stream)
    with (output_path.parent / "summary.md").open("w", encoding="utf-8") as stream:
        stream.write("# CodeFinder Retrieval Evaluation\n\n")
        stream.write(f"Documents: {report['dataset_documents']}  \n")
        stream.write(f"Queries: {report['evaluation_query_count']}  \n")
        stream.write("Judgments: deterministic seed qrels, not expert human judgments.\n\n")
        stream.write("| Mode | NDCG@10 | MAP | Recall@10 | Mean latency (ms) | p95 latency (ms) |\n")
        stream.write("|---|---:|---:|---:|---:|---:|\n")
        for mode, metrics in mode_reports.items():
            stream.write(f"| {mode} | {metrics.get('ndcg@10', 0):.4f} | "
                         f"{metrics.get('map', 0):.4f} | {metrics.get('recall@10', 0):.4f} | "
                         f"{metrics.get('latency_mean_ms', 0):.2f} | "
                         f"{metrics.get('latency_p95_ms', 0):.2f} |\n")
        stream.write("\n## Limitations\n\n")
        for limitation in report["limitations"]:
            stream.write(f"- {limitation}\n")
    return report


def run_graph_analysis() -> dict:
    from .dataset import load_documents
    return graph_report(load_documents())


__all__ = ["run_evaluation", "run_graph_analysis"]
