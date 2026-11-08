# Evaluation report

All numbers below are from the committed run in `data/evaluation/results/`. They were produced by `scripts/evaluate.py` on this corpus and this machine; nothing in this document is a placeholder or an estimate.

**Read the [judgment caveat](#what-these-numbers-are-not) before quoting any metric.** The qrels are deterministic corpus-derived seed judgments, not expert human assessments.

## Setup

| Property | Value |
|---|---|
| Documents | 2,725 |
| Evaluation queries | 40 (27 `natural_question`, 9 `exact_error`, 4 `paraphrase`) |
| Queries with ≥1 positive judgment | 40 |
| Qrel rows | 3,196 (mean 80 judged relevant per query) |
| Languages covered by anchors | 19 |
| Seed questions | 100 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| Metrics | NDCG@5/10, MAP, Recall@5/10, latency mean/p95 |

Every mode is run over the identical query set, qrels, and ranking code, at `top_k=100` with re-ranking enabled, so differences reflect retrieval and fusion rather than harness variation.

## Metrics

| Mode | NDCG@5 | NDCG@10 | MAP | Recall@5 | Recall@10 | Mean (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| tfidf | 0.7779 | 0.7107 | 0.3749 | 0.0911 | 0.1235 | 79.55 | 89.25 |
| bm25 | 0.8454 | 0.7773 | 0.3765 | 0.0954 | 0.1308 | 85.46 | 109.21 |
| semantic | 0.8749 | 0.8117 | 0.4612 | 0.1033 | 0.1501 | 38.91 | 53.36 |
| hybrid_rrf | 0.8907 | 0.8292 | 0.4577 | 0.1049 | **0.1541** | 130.24 | 179.75 |
| hybrid_weighted | **0.8961** | **0.8351** | **0.4782** | **0.1079** | 0.1537 | 131.39 | 145.76 |
| hybrid_prf | 0.8610 | 0.7995 | 0.4609 | 0.1010 | 0.1442 | 276.34 | 414.51 |

Source: `data/evaluation/results/summary.json`. Per-query rankings for every mode are preserved in `rankings.json` so these aggregates can be recomputed or plotted without re-running retrieval.

### Reading the results

BM25 beats TF-IDF on every quality metric (NDCG@10 0.7773 vs 0.7107), which is the expected effect of length normalization and the Robertson–Sparck Jones IDF on a corpus whose documents range from 27 to 3,334 tokens.

Semantic retrieval beats both lexical modes and is also the fastest mode at 38.91 ms mean. That ordering is a property of this setup, not a general claim: lexical search scans postings in Python, while FAISS `IndexFlatIP` over 2,725 normalized 384-d vectors is a single cache-friendly matrix operation. The lexical timings would improve with a compiled postings implementation; the semantic timing excludes one-time model load at startup.

Hybrid fusion gives the best quality overall. Weighted fusion leads every quality metric except Recall@10, where RRF is ahead by 0.0004 — a gap small enough to call a tie. The gains over semantic alone are modest: +0.023 NDCG@10 and +0.017 MAP for weighted fusion. On 40 queries, differences at that scale do not support a significance claim, and the honest summary is that hybrid fusion is at least as good as semantic alone and that the two fusion strategies are equivalent here.

Recall@10 is low in absolute terms (0.12–0.15) for all modes. This is a property of the qrels rather than a retrieval failure: the seed generator marks a mean of 80 documents relevant per query, so ten results cannot recover a large fraction of them. Compare recall across modes, not against 1.0.

## PRF ablation

Measured on the same queries and qrels, toggling only the `prf` flag:

| Variant | NDCG@5 | NDCG@10 | MAP | Recall@10 | Mean (ms) |
|---|---:|---:|---:|---:|---:|
| PRF disabled | 0.8907 | 0.8292 | 0.4577 | 0.1541 | 129.60 |
| PRF enabled | 0.8610 | 0.7995 | 0.4609 | 0.1442 | 276.34 |

**PRF costs ranking quality and roughly doubles latency.** It is reported because it was measured, not because it helped. NDCG@10 drops 0.030, Recall@10 drops 0.010, and mean latency goes from 129.60 ms to 276.34 ms. MAP is the one exception, rising 0.003 — a gap far too small to offset the rest.

The likely cause is query drift. Expansion terms come from the top three lexical documents, and most evaluation queries are already long strings carrying their anchor document's title. Those queries are not term-starved, which is the condition PRF is meant to fix; adding eight expansion terms to an already-specific query dilutes it and pulls in topically similar but non-matching results. The damage is concentrated at the top of the ranking, where NDCG@5 falls 0.030.

A well-founded next step is to make expansion conditional — gate it on query length or on initial-result agreement, so only genuinely underspecified queries get expanded — and re-measure. PRF remains selectable per request rather than removed, since the ablation needs the code path to exist.

## Embedding model comparison

Both models were encoded and evaluated over the same query set and qrels (`embedding_model_comparison.json`):

| Model | Dim | NDCG@5 | NDCG@10 | MAP | Recall@10 | Encode + search (s) |
|---|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | 384 | 0.8468 | 0.7919 | 0.4591 | 0.1533 | 143.92 |
| `paraphrase-MiniLM-L3-v2` | 384 | 0.8021 | 0.7472 | 0.4156 | 0.1343 | 34.66 |

L6 wins on quality across the board (+0.045 NDCG@10, +0.044 MAP); L3 encodes the corpus about 4× faster with its three transformer layers instead of six. L6 is the configured default (`config.DEFAULT_EMBEDDING_MODEL`), since the build cost is paid once and the quality gap is paid on every query.

Both numbers are CPU measurements and both differ from the main table's semantic row, because this script runs its own encode-and-search pass. The encode timings are also not directly comparable to each other: the L6 run included downloading and loading weights, so the 4× ratio overstates the steady-state difference.

## Link graph analysis

`scripts/evaluate.py --graph-only` builds a document graph from shared tags, technology, language, and diagnostic error terms (edge requires ≥2 shared features, capped at 20 neighbors), then computes PageRank and HITS from scratch.

| Property | Value |
|---|---|
| Nodes | 2,725 |
| Directed edges | 54,425 |
| Mean out-degree | 19.97 |

The mean out-degree of 19.97 against a cap of 20 says the graph is saturated: nearly every document hits the neighbor limit. That makes the scores weak discriminators here — the PageRank distribution is close to uniform, and the inspected authority scores are near zero while hub scores are near-identical across documents.

This is the stated reason graph scores are computed and persisted to `data/processed/graph_scores.json` but **not injected into default ranking**. Feeding a near-uniform popularity-like signal into relevance would add bias without adding information. Using it would first require a sparser graph (higher `min_shared`, or weighting by feature rarity) and a separate controlled evaluation.

## Why these numbers are lower than the previous run

Every metric in this report is lower than the run it replaces. No retrieval, ranking, or indexing code changed between the two runs — only the evaluation data generator, which had four defects that inflated the earlier numbers.

| | Previous | Current |
|---|---:|---:|
| Qrel rows | 3,522 | 3,196 |
| Relevance-2 judgments | 1,102 | 383 |
| Judgments for the worst-affected query | 469 | 254 |
| `hybrid_rrf` NDCG@10 | 0.8804 | 0.8292 |

The dominant cause was qrel contamination. Relevance 2 was awarded for sharing a single error term, with no check that the term was specific. Two queries whose anchor error text was the bare label `"error:"` tokenized to just `["error"]`, a token present in 358 documents — those two queries alone accounted for 714 of the 1,102 relevance-2 rows. Documents were being marked relevant for containing the word "error".

The other three defects were in query generation: a length-only guard that admitted bare labels as queries, a query-type selector keyed to list position that never produced paraphrase queries, and a filter too permissive to reject prose sentences that merely mention an error.

**The lower numbers are the more accurate ones.** The previous run measured retrieval against a judgment set that rewarded matching a generic token. The conclusions that survived the correction: BM25 > TF-IDF, semantic > both lexical modes, hybrid ≥ semantic, PRF hurts, L6 > L3. The conclusion that did not: RRF no longer leads NDCG — weighted fusion does, and the two are now effectively tied.

## Metric implementations

From `backend/app/evaluation/metrics.py`, since metric definitions vary across the literature:

- **NDCG@K** — exponential gain `(2^rel - 1) / log2(rank + 1)`, with the ideal DCG computed from all judged relevances for that query, truncated to K. Returns 0.0 when a query has no judged relevant documents.
- **MAP** — average precision over the full returned ranking with relevance defined as `qrel > 0`, divided by the total count of relevant documents (so unretrieved relevant documents correctly depress the score), then macro-averaged.
- **Recall@K** — judged relevant documents in the top K, over all judged relevant documents.
- **Latency** — wall-clock per query around the full `pipeline.search` call, including re-ranking. p95 is linear-interpolated.

Macro-averaging covers only queries with at least one positive judgment; all 40 qualify here, which `judged_queries` records per mode.

## What these numbers are not

- **Not human relevance judgments.** Qrels are generated deterministically: relevance 3 for the query's anchor document, 2 for documents sharing a distinctive error term (one appearing in ≤2% of the corpus), 1 for documents sharing ≥2 query-bearing terms plus language or technology. The generator records `human_judgments: false` in `judgment_provenance.json`.
- **Selection-biased by construction.** Every query is derived from an anchor document in the same corpus it searches, and most queries embed that anchor's title verbatim. This favors lexical matching and guarantees a relevant document exists — the high NDCG values reflect that setup as much as retrieval quality.
- **Query mix still skewed.** 27 of 40 queries are `natural_question`, 9 `exact_error`, 4 `paraphrase`. Better than the previous run, which produced 36/4/0 and two queries that were the literal string `"error:"`, but the per-type counts are still too small to compare types against each other.
- **Error-derived evidence depends on a heuristic field.** Only 502 of 2,725 documents have an `error_message`, and 160 of those are bare labels rejected by `usable_error_text`. Relevance-2 judgments therefore rest on roughly 342 documents, and `extract.py`'s fallback pattern is the upstream reason the field is noisy.
- **Not a significance test.** 40 queries, one machine, one run, no confidence intervals. The spread between the top four modes is under 0.025 NDCG@10, which is within the range these 40 queries cannot resolve — treat the hybrid/semantic ordering as unresolved rather than as a ranking.
- **Not production latency.** CPU-only, local indexes, warm process, single-threaded, no concurrent load.

For a defensible evaluation, judge a held-out query set that was written independently of the corpus, use multiple assessors, and report inter-assessor agreement. The current setup is valid for the thing it is used for here: reproducible A/B comparison of retrieval modes under identical conditions.

## Reproducing

```bash
PYTHONPATH=backend python scripts/prepare_evaluation.py --queries 40 --seeds 100
PYTHONPATH=backend python scripts/evaluate.py
PYTHONPATH=backend python scripts/evaluate.py --graph-only
PYTHONPATH=backend python scripts/compare_embedding_models.py
```

Requires built indexes and embeddings. Quality metrics are deterministic given the same corpus and artifacts; latency figures will differ by machine. The model comparison downloads the second model on first run.

## Related documents

- [Retrieval algorithms](retrieval.md) — what each mode does.
- [Indexing](indexing.md) — index statistics behind these runs.
- [Dataset and attribution](dataset.md) — corpus collection bias.
