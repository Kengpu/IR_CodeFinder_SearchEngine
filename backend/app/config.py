"""CodeFinder configuration.

Central place for paths, tunable IR parameters, and dataset constants.
Every module reads paths from here so the project can be relocated.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Project root = parent of the backend/ directory (this file lives in backend/app)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SEEDS_DIR = DATA_DIR / "seeds"

INDEX_DIR = PROJECT_ROOT / "indexes"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
CONFIG_DIR = PROJECT_ROOT / "config"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

CORPUS_PATH = PROCESSED_DIR / "corpus.jsonl"
STATS_PATH = PROCESSED_DIR / "stats.json"
SOURCES_CSV = DATA_DIR / "sources.csv"

INVERTED_INDEX_PATH = INDEX_DIR / "inverted_index.pkl"
DOC_LENGTHS_PATH = INDEX_DIR / "doc_lengths.pkl"
META_INDEX_PATH = INDEX_DIR / "meta_index.json"
SQLITE_PATH = INDEX_DIR / "metadata.db"

DOC_EMBEDDINGS_PATH = EMBEDDINGS_DIR / "doc_embeddings.npy"
DOC_IDS_PATH = EMBEDDINGS_DIR / "doc_ids.json"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
EMBEDDING_MODEL_PATH = EMBEDDINGS_DIR / "model_name.txt"

LINK_GRAPH_PATH = PROCESSED_DIR / "link_graph.json"
EVAL_DIR = DATA_DIR / "evaluation"
QRELS_PATH = EVAL_DIR / "qrels.tsv"
EVAL_QUERIES_PATH = EVAL_DIR / "eval_queries.jsonl"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# --------------------------------------------------------------------------
# Corpus / crawler settings
# --------------------------------------------------------------------------
# Stack Exchange API: 300 requests/day without an app key, throttled politely.
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"
STACKEXCHANGE_SITE = "stackoverflow"
# Politeness gap between crawler requests (seconds).
CRAWLER_DELAY_SECONDS = float(os.environ.get("CODEFINDER_CRAWLER_DELAY", "1.5"))

TARGET_DOCS = int(os.environ.get("CODEFINDER_TARGET_DOCS", "350"))
TARGET_SEEDS = 100

# --------------------------------------------------------------------------
# Text processing
# --------------------------------------------------------------------------
# Stop-word removal and stemming are switches so experiments can toggle them.
ENABLE_STOPWORDS = True
ENABLE_STEMMING = False  # Porter stemmer destroys code identifiers; off by default
ENABLE_LOWERCASING = True
ENABLE_ACCENT_FOLDING = True

# --------------------------------------------------------------------------
# Lexical retrieval defaults
# --------------------------------------------------------------------------
BM25_K1 = 1.5
BM25_B = 0.75
TFIDF_SMOOTHING = 0.5  # smoothing for idf: log((N - df + 0.5) / (df + 0.5) + 1)

# --------------------------------------------------------------------------
# Semantic retrieval
# --------------------------------------------------------------------------
# Default bi-encoder. Kept small (~80MB) so it downloads fast and encodes a
# few hundred documents in seconds on CPU. Alternatives compared in DEN-08.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32
FAISS_CANDIDATES_MULTIPLIER = 4  # fetch k*m candidates before filtering

# --------------------------------------------------------------------------
# Ranking / fusion defaults
# --------------------------------------------------------------------------
RRF_K = 60          # standard RRF constant
RRF_WINDOW = 100    # only top-N ranks from each searcher participate in fusion
LEXICAL_WEIGHT = 0.55
SEMANTIC_WEIGHT = 0.45
RERANKER_WEIGHT = 0.30  # weight of re-ranker score in final blend, if enabled

# PRF (pseudo-relevance feedback)
PRF_ENABLED = True
PRF_TOP_M = 3       # number of top docs used for expansion
PRF_TERMS = 8       # expansion terms added
PRF_ORIGINAL_WEIGHT = 0.7  # weight of original query vs expanded part in BM25

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
EVAL_K_VALUES = [5, 10]


def ensure_dirs() -> None:
    """Create all output directories (idempotent)."""
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, SEEDS_DIR, INDEX_DIR,
              EMBEDDINGS_DIR, CONFIG_DIR, LOGS_DIR, EVAL_DIR,
              EVAL_RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
