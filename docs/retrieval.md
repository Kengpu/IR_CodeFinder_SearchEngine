# Retrieval algorithms and ranking

## Tokenization and normalization

The tokenizer applies Unicode NFKC and optional accent folding. It preserves code-aware forms: identifiers (including camel-case exceptions), error codes, numbers, paths, punctuation, operators, and common commands. Natural language words are lowercased and can have stopword removal/stemming toggled for experiments. Canonical index keys are case-folded, so `TypeError` can match case-insensitively without losing its display spelling in snippets.

## Inverted index

For each document, the builder tokenizes `title`, error message, problem description, solution, code, and source tags. Each canonical term maps to postings `(doc_id, tf, positions)`. Document frequency is posting-list size; document length is token count; average length supports BM25 normalization. Language, technology, and source-tag metadata maps enable filters.

## TF-IDF

The implementation uses log-scaled term frequency and smoothed inverse document frequency:

```text
TF(t,d)  = 1 + ln(tf(t,d))
IDF(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(q,d) = sum_t TF(t,d) * IDF(t) * qtf(t)
```

The postings implementation scores only documents containing query terms. A configurable smoothing value stabilizes ubiquitous/rare terms.

## BM25

For each query term, the searcher uses positive Robertson–Sparck Jones IDF and standard length normalization:

```text
IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
BM25(t,d) = IDF(t) * tf(t,d)*(k1+1) /
             (tf(t,d) + k1*(1-b + b*|d|/avgdl))
```

Defaults are `k1=1.5`, `b=0.75`. Tests compare posting-based output and scores against an independent document-scan BM25 reference implementation.

## Boolean retrieval

A tokenizer/parser recognizes explicit `AND`, `OR`, `NOT`, and parentheses. Shunting-yard conversion applies `NOT > AND > OR` precedence. Adjacent operands use implicit OR for natural search strings. Operators are uppercase-sensitive, so ordinary prose containing “and” is not silently parsed as Boolean syntax. Matches are filtered by metadata and ordered deterministically by matched positive terms and source score.

## Semantic retrieval and FAISS

The selected bi-encoder is `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional). Documents and queries are L2-normalized. FAISS `IndexFlatIP` over normalized vectors returns cosine similarity. `doc_ids.json` is persisted in exact FAISS row order and validated against vector count and metadata IDs when loaded. For filtered requests, search over-fetches the full index to avoid losing matches outside a small unfiltered candidate window.

A second model (`sentence-transformers/paraphrase-MiniLM-L3-v2`, 384-dimensional) was evaluated on the same query set/qrels. On this CPU run the L6 model had higher NDCG@10/MAP; L3 encoded faster. See `data/evaluation/results/embedding_model_comparison.json`.

## Hybrid retrieval

- **Candidate retrieval**: BM25 and semantic search each produce ranked candidate lists.
- **RRF**: `sum(1/(k + rank))`, one-based rank, default `k=60`.
- **Weighted fusion**: independently min-max-normalizes each candidate list, then combines normalized scores with configurable lexical/semantic weights (default 0.55/0.45).
- **Re-ranking**: transparent field-aware scoring blends normalized base score with title/error overlap and records every component.
- **PRF**: terms are extracted from the top three lexical documents; original terms are kept, common terms excluded, then expanded BM25 is fused with semantic results. PRF is selectable and measured enabled vs disabled.

Each result includes method and score provenance, matched terms/ranks, re-ranker components, final score, text snippet, and highlight terms. Snippet highlighting escapes source/query text before inserting generated `<mark>` wrappers.

## Optional graph analysis

The graph module connects documents sharing source tags, technology, language, or diagnostic error terms. It computes PageRank with dangling-node redistribution and iterative HITS authorities/hubs. Graph scores are persisted for analysis but are **not injected by default**: popularity-like graph features could bias relevance and require a separate controlled evaluation before ranking use.

## Evaluation interpretation

The evaluation runner records NDCG@5/10, MAP, Recall@5/10 and query latency (mean/p95), compares lexical/semantic/hybrid modes, and measures PRF ablation. Current judgments are deterministic corpus-derived seed qrels: the exact anchor, a shared *distinctive* error term (one appearing in at most 2% of the corpus), and shared language/technology evidence. The distinctiveness threshold matters — without it, documents were judged relevant for sharing the generic token `error`. Judgments are explicitly **not expert human judgments**, and queries are anchored to corpus documents, so there is selection bias. See [evaluation.md](evaluation.md) before interpreting the metrics.
