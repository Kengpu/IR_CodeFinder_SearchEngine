# Task status

State of each workstream, with the file or artifact that evidences it. Workstream prefixes match the task IDs in module docstrings (`IDX`, `LEX`, `DEN`, `RNK`, `EVL`).

Last reviewed: 2026-11-08.

## Summary

| Workstream | State | Evidence |
|---|---|---|
| IDX — collection, schema, indexing | Complete | `data/processed/corpus.jsonl`, `indexes/` |
| LEX — TF-IDF, BM25, Boolean | Complete | `backend/app/retrieval/lexical.py` |
| DEN — embeddings, FAISS | Complete | `embeddings/faiss.index` |
| RNK — fusion, PRF, re-ranking | Complete; PRF regresses quality | `backend/app/ranking/fusion.py`, [evaluation](evaluation.md#prf-ablation) |
| API — FastAPI endpoints | Complete | `backend/app/api/routes.py` |
| UI — React SPA | Complete | `frontend/src/` |
| EVL — metrics, ablation, graph | Complete; graph not used in ranking | `data/evaluation/results/` |
| Docs | Complete | `docs/` |

## IDX — data collection and indexing

Done. The Stack Exchange crawler (`crawler/stackexchange.py`) rate-limits, honors the API's returned `backoff`, and preserves raw responses in `data/raw/` so extraction is replayable. Extraction, HTML-to-text conversion, schema validation, and deduplication produce 2,725 documents with zero schema validation errors (`data/processed/stats.json`).

The identifier-preserving tokenizer and the hand-written inverted index are built and persisted: 43,516 terms, 303,628 posting entries, mean document length 366.26 tokens, plus SQLite metadata and language/technology/tag maps. See [indexing](indexing.md).

Known gaps, by design: `error_message` is extracted heuristically and is present on 502 of 2,725 documents; language and technology labels are inferred from source tags, not manually annotated.

## LEX — lexical retrieval

Done. TF-IDF (log TF, smoothed IDF), BM25 (`k1=1.5`, `b=0.75`, configurable), and a Boolean parser with `AND`/`OR`/`NOT`, parentheses, shunting-yard precedence, and implicit OR between adjacent operands. Operators are uppercase-sensitive, so prose containing "and" is not parsed as syntax. BM25 is tested against an independent document-scan reference implementation rather than against itself.

## DEN — semantic retrieval

Done. `all-MiniLM-L6-v2` (384-d) document and query embeddings, L2-normalized, searched through FAISS `IndexFlatIP` so inner product equals cosine similarity. `doc_ids.json` is persisted in FAISS row order and validated against both vector count and metadata IDs at load, which is the failure mode that would otherwise silently return the wrong documents. Filtered requests over-fetch so matches outside a small candidate window are not lost.

The alternative model (`paraphrase-MiniLM-L3-v2`) was evaluated on the same queries and qrels and lost on every quality metric while encoding ~16% faster. L6 stays the default.

## RNK — fusion and ranking

Done, with one negative result worth stating plainly.

RRF (`k=60`, one-based ranks), weighted fusion with independent min-max normalization (0.55 lexical / 0.45 semantic), field-aware re-ranking, duplicate suppression, and per-result ranking explanations are all implemented. Hybrid modes give the best measured quality: weighted fusion leads NDCG@10 (0.8351) and MAP (0.4782), with RRF ahead on Recall@10 by 0.0004 — effectively a tie between the two strategies.

**PRF is implemented but regresses quality.** With expansion enabled, NDCG@10 drops from 0.8292 to 0.7995 and mean latency roughly doubles (129.60 ms → 276.34 ms). The probable cause is query drift: most evaluation queries are already long natural-language strings, so they are not term-starved and expansion only dilutes them. It stays selectable per request rather than removed, because the ablation needs the code path. The open follow-up is to gate expansion on query length or initial-result agreement, then re-measure. See [evaluation](evaluation.md#prf-ablation).

## API and UI

Done. Nine endpoints: `/health`, `/search`, `/documents/{id}`, `/similar/{id}`, `/languages`, `/technologies`, `/evaluation`, plus Swagger at `/docs`. Contract documented in [api_contract.md](api_contract.md).

The React + TypeScript + Vite SPA covers search modes, filters, loading/error/empty states, document detail, code copy, source attribution, ranking explanation, related errors, and the evaluation table. Results are never hardcoded: every search, detail, and related-error view calls the backend.

## EVL — evaluation

Done, measured, and reported with its limitations in [evaluation.md](evaluation.md). 40 queries, 3,196 qrel rows, six retrieval modes, PRF ablation, NDCG@5/10, MAP, Recall@5/10, and mean/p95 latency. Full per-query rankings are preserved in `rankings.json` so aggregates can be recomputed without re-running retrieval.

The query/qrel generator was corrected on 2026-11-08 after four defects were found that inflated the earlier metrics — most significantly, relevance-2 judgments awarded for sharing the generic token `error`. All reported numbers are from the post-fix run; see [why these numbers are lower](evaluation.md#why-these-numbers-are-lower-than-the-previous-run).

PageRank and HITS are computed from scratch over a 2,725-node, 54,425-edge document graph and persisted to `data/processed/graph_scores.json`. **They are deliberately not injected into ranking.** Mean out-degree is 19.97 against a neighbor cap of 20, so the graph is saturated and the scores are near-uniform — a weak discriminator that would add popularity bias without adding information. Using it would require a sparser graph and a separate controlled evaluation.

The qrels are the main caveat on everything in this section: deterministic corpus-derived seed judgments with `human_judgments: false`, and queries anchored to documents in the corpus they search. Valid for A/B comparison between modes under identical conditions; not a relevance study.

## Verification

29 backend test functions across five files (`test_api.py` 6, `test_lexical.py` 10, `test_semantic.py` 5, `test_fusion.py` 4, `test_evaluation.py` 4) and 3 frontend tests covering query submission, live result rendering, and the backend-failure state. GitHub Actions runs backend `pytest -q` and frontend `npm ci && npm run build && npm test` on push and pull request (`.github/workflows/ci.yml`).

Counts above are from static inspection of the test files. The suites were last run as recorded in the README; no `.venv` exists in this working tree, so they were not re-executed during this review. To verify:

```bash
cd backend && pytest -q
cd ../frontend && npm test && npm run build
```

## Open items

Ordered by value, none blocking.

1. **Make PRF conditional and re-measure.** The only known quality regression in the system. Gate expansion on query length or initial-result agreement so only underspecified queries get expanded.
2. **Independent evaluation set.** Queries written without reference to the corpus, judged by multiple assessors with reported agreement. This is what would turn the current metrics into a defensible relevance claim.
3. **Tighten `_extract_error_message` upstream.** `backend/app/crawler/extract.py:156` falls back to the pattern `\berror\b[^\n]{0,200}`, which captures the word "error" anywhere in prose plus whatever follows. That is why 160 of 502 populated `error_message` fields are bare labels or sentences merely mentioning an error. `usable_error_text` now filters these at the evaluation layer, but the field is also used for snippets, field-aware re-ranking, and the graph, so the real fix belongs in extraction. Changing it requires rebuilding the corpus, indexes, and embeddings.
4. **Sparser document graph.** Raise `min_shared` or weight edges by feature rarity so PageRank and HITS discriminate; only then consider them as bounded tie-break features.
5. **`npm audit` findings.** The current dependency tree reports audit findings and no forced upgrade was applied. Review before any deployment.
6. **Error extraction coverage.** Only 502 of 2,725 documents carry an `error_message` at all, and 342 survive filtering. Since both the qrel error-evidence rule and the exact-error query template depend on that field, better extraction would widen evaluation coverage as well as re-ranking quality.

## Resolved

**Evaluation query/qrel generator defects** (fixed 2026-11-08, `backend/app/evaluation/dataset.py`). Four defects, each with a regression test:

1. *Qrel contamination.* Relevance 2 required sharing only one error term, with no specificity check. Two queries anchored to the bare label `"error:"` tokenized to `["error"]`, a token in 358 documents, producing 714 of 1,102 relevance-2 rows. Now gated by `distinctive_error_terms`, which requires a term to appear in ≤2% of the corpus. Relevance-2 rows fell to 383.
2. *Degenerate queries.* The `len(error) >= 5` guard admitted `"error:"` as a query. Replaced by `usable_error_text`, which keeps short code-like diagnostics (`TypeError`, `ENOENT`) and rejects bare labels and prose sentences about errors.
3. *Unreachable paraphrase branch.* Query type was chosen by a document's index in the full list, so the paraphrase template never fired. It now keys off a counter over eligible documents; the set went from 36/4/0 to 27/9/4 across the three types, covering all 19 languages.
4. *Quadratic qrel construction.* `_terms(candidate)` was recomputed per (query, document) pair, tokenizing the whole corpus once per query. Term sets are now precomputed once. Anchor membership also used `doc not in selected`, comparing whole Pydantic models; it now checks IDs.

All metrics were re-measured after the fix and are lower, because the previous run was scored against inflated judgments.

## Out of scope by design

No authentication, registration, owner IDs, or ownership validation. CORS is restricted to local development origins. Pickle artifacts are trusted local builds and are never loaded from uploads.
