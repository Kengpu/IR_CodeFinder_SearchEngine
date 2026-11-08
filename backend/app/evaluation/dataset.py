"""Evaluation data and qrel utilities (EVL-01..03).

Evaluation queries are stored as JSONL records. Qrels use the standard TSV
format: query_id, iteration, document_id, relevance. The generator anchors
each query to a real corpus document and records the provenance in the JSONL
record; it never creates document IDs or source URLs that are absent from the
corpus.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .. import config
from ..indexing.tokenizer import canonical_token, tokenize
from ..models.document import Document


# Heuristic error extraction leaves bare labels such as "error:" in the corpus
# (160 of 502 populated fields in the current snapshot). They are not usable as
# queries and not usable as relevance evidence, so both uses share one filter.
_GENERIC_ERROR_TEXT = frozenset({
    "error", "errors", "exception", "exceptions", "error message", "error messages",
    "error is", "error of", "errors are", "the error", "an error", "this error",
    "error occurred", "traceback (most recent call last)", "stack trace",
    "output", "result", "problem", "issue", "warning", "warnings",
})
# Prefixes that mark the start of a prose sentence about an error rather than
# the error itself. `extract.py` falls back to a bare ``\berror\b`` pattern, so
# values like "error when compiling in Swift." reach the corpus.
_GENERIC_PREFIXES = ("error", "errors", "exception", "warning", "traceback",
                     "error message", "error-prone", "stack trace")

# A term shared by more than this fraction of the corpus carries no evidence of
# a shared diagnostic, so it must not promote a document to relevance 2.
MAX_ERROR_TERM_DOCUMENT_RATIO = 0.02

_WORD_RE = re.compile(r"[A-Za-z]+")
_TOKEN_SPLIT_RE = re.compile(r"[\s,;()\[\]{}'\"`]+")


def _is_code_token(token: str) -> bool:
    """True when a single token looks like a code identifier.

    Checked per token rather than per string: scanning a whole string for any
    uppercase character treats an ordinary capitalized word ("Swift") as an
    identifier, which is exactly how prose fragments slip through.
    """
    token = token.strip(".:?!,")
    if len(token) < 2:
        return False
    return (any(c.isupper() for c in token[1:])      # TypeError, ENOENT
            or any(c.isdigit() for c in token)        # HTTP 500, 2.11.1
            or "_" in token                           # E_ALL, ERR_FOO
            or ("." in token[1:-1] and any(c.isalpha() for c in token)))  # javaw.exe


def _has_code_token(text: str) -> bool:
    return any(_is_code_token(t) for t in _TOKEN_SPLIT_RE.split(text) if t)


def usable_error_text(text: str | None) -> str:
    """Return trimmed error text that is specific enough to use, else "".

    Keeps code-like diagnostics (``TypeError``, ``ENOENT``, ``HTTP 500``) even
    when short, since an exception name alone is a precise query, and keeps
    descriptive diagnostics that carry no identifier ("cannot find module
    react"). Rejects bare labels and prose sentences merely *about* an error.
    """
    error = (text or "").strip().strip("`\"'").strip()
    if not error:
        return ""
    if error.strip(" :?!.,").casefold() in _GENERIC_ERROR_TEXT:
        return ""
    # A value that opens with a generic label is only usable when the remainder
    # carries a real identifier ("error: Untracked ... 'example.txt'" is an
    # error; "error when compiling in Swift." is a sentence about one).
    lead = _WORD_RE.match(error.casefold())
    first_word = error[:lead.end()].casefold() if lead else ""
    if first_word in _GENERIC_PREFIXES or error.casefold().startswith("error-prone"):
        remainder = error[len(first_word):]
        return error if _has_code_token(remainder) and len(error) >= 12 else ""
    if _has_code_token(error) and len(error) >= 5:
        return error
    # Allow descriptive prose only when it is long enough to be specific.
    words = [w for w in _WORD_RE.findall(error) if len(w) >= 2]
    if len(words) >= 3 and len(error) >= 15:
        return error
    return ""


DEFAULT_SEED_TEMPLATES = [
    ("Why does {error} happen in {language}?", "exact", "{language}"),
    ("How do I fix {error} when working with {technology}?", "exact", "{technology}"),
    ("{language} program reports {error}; what is the solution?", "exact", "{language}"),
    ("What causes this {technology} error and how can I resolve it?", "paraphrase", "{technology}"),
    ("How can I debug a {language} application that is not working?", "paraphrase", "{language}"),
    ("Show a safe fix for a common {technology} exception", "paraphrase", "{technology}"),
    ("What does this programming error message mean?", "natural", "general"),
    ("How do I troubleshoot a failed developer command?", "natural", "general"),
    ("Why is my code returning an unexpected result?", "natural", "general"),
    ("How can I resolve an invalid configuration in my project?", "natural", "general"),
]


def load_documents(path: Path | None = None) -> list[Document]:
    path = path or config.CORPUS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return [Document.model_validate_json(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def create_seed_queries(docs: list[Document], count: int = config.TARGET_SEEDS) -> list[dict]:
    """Create a broad, auditable seed-query set from corpus vocabulary.

    The templates are question-writing prompts rather than search results;
    their language/technology values are drawn from observed corpus metadata.
    """
    languages = sorted({d.language for d in docs if d.language})
    technologies = sorted({t for d in docs for t in d.technologies})
    # Bare labels such as "error:" produce meaningless seed questions.
    errors = [text for d in docs if (text := usable_error_text(d.error_message))]
    rows: list[dict] = []
    serial = 1
    # Exact error questions first; these are useful for lexical evaluation.
    for error in errors:
        language = languages[(serial - 1) % len(languages)] if languages else "programming"
        technology = technologies[(serial - 1) % len(technologies)] if technologies else "software"
        template, query_type, _ = DEFAULT_SEED_TEMPLATES[(serial - 1) % len(DEFAULT_SEED_TEMPLATES)]
        rows.append({
            "query_id": f"seed-{serial:03d}",
            "query": template.format(error=error, language=language, technology=technology),
            "query_type": query_type,
            "language": language,
            "technology": technology,
            "error_identifier": error,
            "provenance": "template instantiated from observed corpus error text",
        })
        serial += 1
        if len(rows) >= count:
            return rows
    # Fill remaining coverage with technology/language natural-language prompts.
    dimensions = [("language", x) for x in languages] + [("technology", x) for x in technologies]
    if not dimensions:
        dimensions = [("language", "programming")]
    while len(rows) < count:
        dimension, value = dimensions[(len(rows)) % len(dimensions)]
        template, query_type, _ = DEFAULT_SEED_TEMPLATES[len(rows) % len(DEFAULT_SEED_TEMPLATES)]
        rows.append({
            "query_id": f"seed-{len(rows) + 1:03d}",
            "query": template.format(error="an unexpected error", language=value,
                                      technology=value),
            "query_type": query_type,
            dimension: value,
            "provenance": "template instantiated from observed corpus metadata",
        })
    return rows


def _query_for_doc(doc: Document, error_rank: int) -> tuple[str, str]:
    """Return the evaluation query and type for `doc`.

    `error_rank` counts documents that actually carry usable error text, so the
    exact/paraphrase split is decided among eligible documents. Deciding it from
    the document's position in the full list silently starved the paraphrase
    template: eligible documents all landed on positions it never selected.
    """
    error = usable_error_text(doc.error_message)
    language = doc.language or "programming"
    tech = doc.technologies[0] if doc.technologies else language
    if error:
        # Every third eligible document becomes a paraphrase so all three query
        # types appear whenever enough eligible documents are selected.
        if error_rank % 3 == 0:
            return f"How do I fix the {language} problem reported as {error}?", "paraphrase"
        return error, "exact_error"
    return f"How can I solve this {tech} problem: {doc.title}?", "natural_question"


def create_evaluation_queries(docs: list[Document], count: int = 40) -> list[dict]:
    """Select up to `count` diverse real documents as evaluation anchors.

    Selection balances two goals: cover as many languages as possible, and keep
    enough documents carrying usable error text that exact-error and paraphrase
    queries are both represented rather than left to chance.
    """
    ordered = sorted(docs, key=lambda d: (d.language or "", d.doc_id))
    selected: list[Document] = []
    selected_ids: set[str] = set()
    seen_languages: set[str] = set()

    def take(doc: Document) -> None:
        selected.append(doc)
        selected_ids.add(doc.doc_id)
        if doc.language:
            seen_languages.add(doc.language)

    # First pass: one document per language, preferring anchors with usable
    # error text so language coverage and error coverage are not in conflict.
    by_language: dict[str, list[Document]] = {}
    for doc in ordered:
        if doc.language:
            by_language.setdefault(doc.language, []).append(doc)
    for language in sorted(by_language):
        if len(selected) >= count:
            break
        candidates = by_language[language]
        with_error = [d for d in candidates if usable_error_text(d.error_message)]
        take(with_error[0] if with_error else candidates[0])

    # Second pass: aim for roughly a third of anchors to carry error text, so
    # exact_error and paraphrase queries are both generated.
    target_with_error = max(1, count // 3)
    have_with_error = sum(1 for d in selected if usable_error_text(d.error_message))
    if have_with_error < target_with_error:
        for doc in ordered:
            if len(selected) >= count or have_with_error >= target_with_error:
                break
            if doc.doc_id not in selected_ids and usable_error_text(doc.error_message):
                take(doc)
                have_with_error += 1

    # Fill any remainder. Membership is checked by ID: ``doc not in selected``
    # compared whole Pydantic models, which is both O(n^2) and equality-dependent.
    for doc in ordered:
        if len(selected) >= count:
            break
        if doc.doc_id not in selected_ids:
            take(doc)

    rows = []
    error_rank = 0
    for number, doc in enumerate(selected, start=1):
        if usable_error_text(doc.error_message):
            error_rank += 1
        query, query_type = _query_for_doc(doc, error_rank)
        rows.append({
            "query_id": f"q{number:03d}",
            "query": query,
            "query_type": query_type,
            "anchor_doc_id": doc.doc_id,
            "language": doc.language,
            "technologies": doc.technologies,
            "source_url": doc.source_url,
            "provenance": "anchor document selected from processed Stack Exchange corpus",
        })
    return rows


def _terms(doc: Document) -> set[str]:
    text = " ".join([doc.title, doc.error_message or "", doc.description])
    return {canonical_token(t) for t in tokenize(text, remove_stopwords=True)
            if len(t) >= 3 and any(c.isalpha() for c in t)}


def _error_terms(doc: Document) -> set[str]:
    """Diagnostic terms from usable error text only.

    Documents whose error field is a bare label contribute nothing, so they
    neither gain nor confer relevance through the error rule.
    """
    error = usable_error_text(doc.error_message)
    if not error:
        return set()
    return {canonical_token(t) for t in tokenize(error, remove_stopwords=False)
            if len(t) >= 3 and any(c.isalpha() for c in t)}


def distinctive_error_terms(documents: list[Document],
                            max_ratio: float = MAX_ERROR_TERM_DOCUMENT_RATIO) -> set[str]:
    """Error terms rare enough to evidence a genuinely shared diagnostic.

    Without this, a term like ``error`` matched hundreds of documents and
    promoted them all to relevance 2, which inflated the judgment set and the
    measured scores without indicating any shared root cause.
    """
    if not documents:
        return set()
    counts: Counter[str] = Counter()
    for doc in documents:
        counts.update(_error_terms(doc))
    # Floor of 2: a term shared by exactly two documents is the smallest
    # meaningful sharing, so a limit of 1 would reject every shared term and
    # disable the relevance-2 rule entirely on small corpora.
    limit = max(2, int(len(documents) * max_ratio))
    return {term for term, count in counts.items() if count <= limit}


def create_qrels(queries: list[dict], documents: list[Document]) -> list[dict]:
    """Create deterministic seed judgments anchored to real documents.

    Relevance levels:
    - 3: the query's known anchor document;
    - 2: another document sharing a *distinctive* error identifier, meaning a
      diagnostic term rare across the corpus (see `distinctive_error_terms`);
    - 1: a document sharing at least two query-bearing terms and language or
      technology metadata.

    These are *seed judgments*, not a claim of expert human assessment; the
    qrel provenance is included in the output and documented in EVL-02.
    """
    by_id = {d.doc_id: d for d in documents}
    # Precompute per-document term sets: these are reused for every query, and
    # recomputing them per (query, document) pair tokenized the whole corpus
    # once per query.
    doc_terms = {d.doc_id: _terms(d) for d in documents}
    doc_error_terms = {d.doc_id: _error_terms(d) for d in documents}
    distinctive = distinctive_error_terms(documents)

    output: list[dict] = []
    for query in queries:
        anchor = by_id.get(query.get("anchor_doc_id", ""))
        if anchor is None:
            continue
        anchor_terms = doc_terms[anchor.doc_id]
        anchor_technologies = set(anchor.technologies)
        # Only distinctive terms may confer relevance 2.
        error_terms = doc_error_terms[anchor.doc_id] & distinctive
        for candidate in documents:
            relevance = 0
            if candidate.doc_id == anchor.doc_id:
                relevance = 3
            else:
                overlap = len(anchor_terms & doc_terms[candidate.doc_id])
                same_meta = bool(
                    (anchor.language and candidate.language == anchor.language)
                    or (anchor_technologies & set(candidate.technologies)))
                if error_terms & doc_error_terms[candidate.doc_id]:
                    relevance = 2
                elif same_meta and overlap >= 2:
                    relevance = 1
            if relevance:
                output.append({
                    "query_id": query["query_id"], "iteration": "0",
                    "doc_id": candidate.doc_id, "relevance": relevance,
                    "provenance": "deterministic seed judgment; anchor plus shared error/metadata evidence",
                })
    return output


def save_qrels(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["query_id", "iteration", "doc_id", "relevance"])
        for row in rows:
            writer.writerow([row["query_id"], row.get("iteration", "0"),
                             row["doc_id"], row["relevance"]])


def load_qrels(path: Path | None = None) -> dict[str, dict[str, int]]:
    path = path or config.QRELS_PATH
    if not path.exists():
        return {}
    output: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            output.setdefault(row["query_id"], {})[row["doc_id"]] = int(row["relevance"])
    return output


def validate_qrels(qrels: dict[str, dict[str, int]], doc_ids: set[str]) -> list[str]:
    errors: list[str] = []
    for query_id, judgments in qrels.items():
        if not judgments:
            errors.append(f"{query_id}: no judgments")
        for doc_id, relevance in judgments.items():
            if doc_id not in doc_ids:
                errors.append(f"{query_id}: unknown document {doc_id}")
            if relevance < 0 or relevance > 3:
                errors.append(f"{query_id}/{doc_id}: relevance outside 0..3")
    return errors


def judgment_guidelines() -> str:
    return ("Use relevance 3 for a directly solving document, 2 for a document "
            "with the same diagnostic error, 1 for a useful related solution, "
            "and 0/unjudged for unrelated results. Record the judge and evidence.")


__all__ = ["load_documents", "create_seed_queries", "create_evaluation_queries",
           "create_qrels", "save_qrels", "load_qrels", "validate_qrels",
           "judgment_guidelines", "write_jsonl", "read_jsonl",
           "usable_error_text", "distinctive_error_terms",
           "MAX_ERROR_TERM_DOCUMENT_RATIO"]
