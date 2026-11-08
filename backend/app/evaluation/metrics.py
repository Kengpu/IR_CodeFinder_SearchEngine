"""IR evaluation metrics (EVL-04): NDCG@K, MAP, and Recall@K."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg(relevances: Sequence[float], k: int | None = None) -> float:
    values = relevances if k is None else relevances[:k]
    return sum((2.0 ** rel - 1.0) / math.log2(rank + 1)
               for rank, rel in enumerate(values, start=1))


def ndcg_at_k(ranked_doc_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    """Normalized discounted cumulative gain; zero if no judged relevant docs."""
    if k <= 0:
        raise ValueError("k must be positive")
    actual = [float(qrels.get(doc_id, 0.0)) for doc_id in ranked_doc_ids[:k]]
    ideal = sorted((float(v) for v in qrels.values()), reverse=True)[:k]
    denominator = dcg(ideal)
    return dcg(actual) / denominator if denominator else 0.0


def average_precision(ranked_doc_ids: Sequence[str], qrels: Mapping[str, float]) -> float:
    """Average precision with relevance defined as qrel > 0."""
    relevant_count = sum(1 for rel in qrels.values() if rel > 0)
    if not relevant_count:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if qrels.get(doc_id, 0.0) > 0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / relevant_count


def recall_at_k(ranked_doc_ids: Sequence[str], qrels: Mapping[str, float], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    relevant = {doc_id for doc_id, rel in qrels.items() if rel > 0}
    if not relevant:
        return 0.0
    return len(set(ranked_doc_ids[:k]) & relevant) / len(relevant)


def evaluate_query_set(results: Mapping[str, Sequence[str]],
                       qrels: Mapping[str, Mapping[str, float]],
                       k_values: Sequence[int] = (5, 10)) -> dict[str, float]:
    """Macro-average metrics over queries with at least one positive judgment."""
    query_ids = [qid for qid, judgments in qrels.items()
                 if any(rel > 0 for rel in judgments.values())]
    if not query_ids:
        return {metric: 0.0 for k in k_values
                for metric in (f"ndcg@{k}", f"recall@{k}")} | {"map": 0.0}
    output: dict[str, float] = {}
    for k in k_values:
        output[f"ndcg@{k}"] = sum(ndcg_at_k(results.get(qid, []), qrels[qid], k)
                                  for qid in query_ids) / len(query_ids)
        output[f"recall@{k}"] = sum(recall_at_k(results.get(qid, []), qrels[qid], k)
                                    for qid in query_ids) / len(query_ids)
    output["map"] = sum(average_precision(results.get(qid, []), qrels[qid])
                        for qid in query_ids) / len(query_ids)
    output["judged_queries"] = float(len(query_ids))
    return output


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile; p is in [0, 100]."""
    if not values:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError("p must be between 0 and 100")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * p / 100
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
