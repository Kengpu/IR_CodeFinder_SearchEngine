# CodeFinder API Contract

Base URL in development: `http://127.0.0.1:8000`

Interactive OpenAPI documentation is available at `/docs`; ReDoc is at `/redoc`.
All endpoints are unauthenticated. The system intentionally performs no owner or resource ownership checks.

## `GET /health`

Returns service/index status.

```json
{
  "status": "ok",
  "lexical_index": true,
  "semantic_index": true,
  "documents": 2725,
  "version": "1.0.0"
}
```

`status` is `degraded` with HTTP 200 when the lexical index is unavailable; the detail explains the startup issue.

## `GET /search`

Required query parameter:

- `q`: non-blank error text/question, max 2,000 characters.

Optional parameters:

| Parameter | Values/default | Meaning |
|---|---|---|
| `mode` | `hybrid_rrf` | `tfidf`, `bm25`, `boolean`, `semantic`, `hybrid_rrf`, `hybrid_weighted`, `hybrid_prf` |
| `top_k` | `10` | 1–100 |
| `language` | omitted | Exact language filter |
| `technology` | omitted | Technology filter |
| `prf` | omitted | Enable/disable PRF for `hybrid_prf` |
| `rerank` | `true` | Enable transparent field re-ranking |
| `lexical_weight` | `0.55` | Weighted-fusion lexical contribution |
| `semantic_weight` | `0.45` | Weighted-fusion semantic contribution |

Example: `/search?q=TypeError%20pandas&mode=hybrid_rrf&language=python&top_k=5`

Response fields:

```json
{
  "query": "TypeError pandas",
  "mode": "hybrid_rrf",
  "total": 1,
  "metadata": {"lexical_candidates": 20, "semantic_candidates": 20, "result_count": 1},
  "results": [{
    "doc_id": "so-123",
    "title": "...",
    "error_message": "TypeError: ...",
    "description": "...",
    "solution": "...",
    "code_example": "...",
    "tags": ["python", "pandas"],
    "language": "python",
    "technologies": ["pandas"],
    "source_url": "https://stackoverflow.com/questions/123",
    "source_site": "stackoverflow",
    "score": 0.73,
    "ranking_explanation": {"method": "rrf", "reason": "..."},
    "snippet": "...",
    "highlighted_snippet": "...<mark>TypeError</mark>..."
  }]
}
```

The highlighted field contains only escaped text plus generated `<mark>` tags. Query failures return 422; missing semantic artifacts for semantic/hybrid modes return 503; an unknown/invalid mode returns 422.

## `GET /documents/{document_id}`

Returns the complete normalized document plus `related_errors`, which are real nearest-neighbor results. Unknown IDs return 404.

## `GET /similar/{document_id}`

Returns up to `top_k` (default 5, max 20) semantically similar documents. Optional `language` and `technology` filters are supported. Unknown IDs return 404.

## `GET /technologies` and `GET /languages`

Returns live corpus facets ordered by descending document count:

```json
[{"name": "python", "count": 523}]
```

## `GET /evaluation`

Returns the saved evaluation run only. Before `scripts/evaluate.py` has run:

```json
{"available": false, "metrics": {}, "message": "Evaluation has not been run yet..."}
```

Afterward it returns measured metrics from `data/evaluation/results/summary.json`. It never creates or estimates metrics in the API.

## Error format

FastAPI validation and application errors use the standard shape:

```json
{"detail": "human-readable explanation"}
```

## CORS

Development browser origins are limited to Vite on `localhost`/`127.0.0.1` ports 5173 and 4173. No credentials are enabled.
