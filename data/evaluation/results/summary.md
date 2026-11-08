# CodeFinder Retrieval Evaluation

Documents: 2725  
Queries: 40  
Judgments: deterministic seed qrels, not expert human judgments.

| Mode | NDCG@10 | MAP | Recall@10 | Mean latency (ms) | p95 latency (ms) |
|---|---:|---:|---:|---:|---:|
| tfidf | 0.7107 | 0.3749 | 0.1235 | 79.55 | 89.25 |
| bm25 | 0.7773 | 0.3765 | 0.1308 | 85.46 | 109.21 |
| semantic | 0.8117 | 0.4612 | 0.1501 | 38.91 | 53.36 |
| hybrid_rrf | 0.8292 | 0.4577 | 0.1541 | 130.24 | 179.75 |
| hybrid_weighted | 0.8351 | 0.4782 | 0.1537 | 131.39 | 145.76 |
| hybrid_prf | 0.7995 | 0.4609 | 0.1442 | 276.34 | 414.51 |

## Limitations

- Seed qrels are deterministic corpus-derived judgments and must not be interpreted as human relevance assessments.
- Evaluation queries include anchor-derived text; this can favor lexical matching and creates a known selection bias.
- For formal claims, independently judge a held-out query set with multiple assessors.
