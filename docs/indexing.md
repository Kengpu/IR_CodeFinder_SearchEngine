# Indexing and text processing

This document describes how processed documents become searchable artifacts. The builder is `backend/app/indexing/`, driven by `scripts/build_index.py`.

## Why a hand-written index

The index is built from scratch instead of delegating to a search library. Each term maps to explicit postings holding term frequency and positions, so TF, DF, TF-IDF, BM25, and Boolean retrieval can be recomputed, inspected, and defended in the project report. The trade-off is accepted deliberately: no library-level query cache, compression, or distributed sharding, in exchange for a transparent and auditable scoring path.

## Tokenization

`backend/app/indexing/tokenizer.py` tokenizes prose and source code with one regular expression whose alternatives are ordered longest-form-first, so specialized forms match before the generic identifier rule.

| Group | Keeps intact | Example |
|---|---|---|
| `winpath` | Windows paths | `C:\Users\app\main.py` |
| `unixpath` | POSIX paths | `/usr/local/bin/node` |
| `command` | Developer commands | `npm install`, `pip install` |
| `errorcode` | 3–5 digit codes | `404`, `HTTP 500`, `ERR_404` |
| `operator` | Code operators | `===`, `=>`, `::`, `&&`, `**` |
| `identifier` | Dotted/hyphenated names | `TypeError`, `node.js`, `my-module` |
| `number` | Versions and numerics | `3.12`, `1e-10` |
| `punct` | Code punctuation | `(`, `{`, `<`, `/` |

Normalization applies Unicode NFKC, optional accent folding, and translation of typographic quotes and dashes that commonly appear in pasted questions. Isolated unsupported symbols (the `other` group) are dropped.

### Case, stopwords, and stemming

`_is_identifier` decides whether a token is code-like: an internal capital, any digit, any of `_$.-/\`, or an `http`/`err_` prefix. A leading capital alone is treated as prose, so `Python` folds to lowercase while `TypeError` keeps its spelling.

Three independent switches in `backend/app/config.py` control the rest, so ablations change one variable at a time:

- `ENABLE_STOPWORDS = True` — a compact English stop list, applied only to plain alphabetic `identifier` tokens that are not code-like. `not` as prose is removed; `NOT` as a Boolean operator and `null` as an identifier are not.
- `ENABLE_STEMMING = False` — off by default because the Porter stemmer mangles code identifiers. When enabled it is restricted to alphabetic prose words.
- `ENABLE_ACCENT_FOLDING = True`.

Multiword commands are indexed twice: as component tokens (`npm`, `install`) and as a joined phrase token (`npm_install`), so both exact and broad queries match.

Index keys are case-folded by `canonical_token`, which changes the key without changing token boundaries. `TypeError` and `typeerror` therefore hit the same postings while snippets still show the original spelling.

## Index structure

`build_index` tokenizes each document's `searchable_text()` — title, error message, description, solution, code, and source tags — then records:

- `postings`: canonical term → doc ID → `Posting(tf, positions)`. `Counter` captures TF in one pass; a second pass records positions for snippets and future phrase search.
- `doc_lengths`: token count per document, used for BM25 length normalization.
- `documents`: the full JSON payload per document, for result rendering.
- `metadata`: case-folded `language`, `technology`, and `tag` maps to document-ID sets, used by filters.
- `total_docs`, `vocabulary_size`, `avg_doc_length`.

Document frequency is derived, not stored: `document_frequency` reads each posting list's length, so DF cannot drift out of sync with the postings.

`filtered_doc_ids` intersects the requested metadata sets and returns `None` when no filter was requested, which lets callers distinguish "no filter" from "filter matched nothing".

## Current build

Measured from the committed artifacts (`PYTHONPATH=backend`, unpickling `indexes/inverted_index.pkl`):

| Property | Value |
|---|---|
| Documents | 2,725 |
| Vocabulary (distinct terms) | 43,516 |
| Posting entries | 303,628 |
| Mean document length | 366.26 tokens |
| Shortest / longest document | 27 / 3,334 tokens |

Artifact sizes: `inverted_index.pkl` 11 MB, `meta_index.json` 8.7 MB, `metadata.db` 8.4 MB, `doc_lengths.pkl` 44 KB.

## Persistence

`InvertedIndex.save()` writes four artifacts, with paths resolved from `config.py` so the project can be relocated:

- `indexes/inverted_index.pkl` — postings plus corpus statistics (highest pickle protocol).
- `indexes/doc_lengths.pkl` — per-document token counts.
- `indexes/meta_index.json` — document payloads and metadata maps, with ID sets serialized as sorted lists for stable diffs.
- `indexes/metadata.db` — SQLite `documents` table backing detail lookups, with an index on `language`.

Two implementation details are deliberate. The SQLite writer closes its connection in a `finally` block, because `sqlite3.Connection.__exit__` commits but does not close, and a leaked handle blocks file replacement on Windows. And `load()` accepts postings stored as plain dicts as well as `Posting` objects, so indexes built by an older revision still load.

`load()` raises `FileNotFoundError` with the rebuild command when any artifact is missing, rather than silently serving an empty index.

## Rebuilding

```bash
PYTHONPATH=backend python scripts/build_index.py
```

Both pickle files are trusted local build artifacts. Unpickling executes code, so never load an index file received from an untrusted source — rebuild it from `data/processed/corpus.jsonl` instead. Note that reading these pickles outside the app requires `PYTHONPATH=backend`, since the postings reference the `app.indexing.inverted.Posting` class.

## Related documents

- [Dataset and attribution](dataset.md) — how `corpus.jsonl` is produced.
- [Retrieval algorithms](retrieval.md) — how the postings are scored.
- [Architecture](architecture.md) — where indexing sits in the pipeline.
