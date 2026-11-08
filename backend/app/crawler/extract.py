"""Data extraction and schema mapping (IDX-04) + cleaning/dedup (IDX-06).

Turns raw Stack Exchange API question/answer JSON into shared `Document`
records, applying:
- HTML -> text conversion (code blocks preserved)
- language/technology tagging from the tag query that found the question
  and from the question's own tags
- exact and near-duplicate removal (normalized-text hashing + MinHash-style
  shingle Jaccard for near-dupes)
- field validation via the pydantic schema

Output: data/processed/corpus.jsonl + data/processed/stats.json (IDX-07).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from .. import config
from ..models.document import Document, validate_documents
from ..crawler.stackexchange import TAG_QUERIES
from ..utils.text import extract_code_blocks, html_to_text

# Map Stack Exchange tag -> (language, technologies). Built from TAG_QUERIES
# plus well-known aliases so documents found under any alias still tag right.
_TAG_TO_META: dict[str, tuple[str, list[str]]] = {}
for _tag, _lang, _techs in TAG_QUERIES:
    _TAG_TO_META[_tag] = (_lang or "", list(_techs))
for _alias, _meta in {
    "reactjs": ("javascript", ["react"]), "react": ("javascript", ["react"]),
    "vuejs": ("javascript", ["vue.js"]), "vue": ("javascript", ["vue.js"]),
    "nodejs": ("javascript", ["node.js"]),
    "postgres": ("sql", ["postgresql"]), "pgsql": ("sql", ["postgresql"]),
    "c#": ("c#", []), "csharp": ("c#", []),
    "python-3.x": ("python", []), "javascript": ("javascript", []),
    "android": ("java", ["android"]), "kotlin": ("kotlin", ["android"]),
    "spring": ("java", ["spring-boot"]), "spring-boot": ("java", ["spring-boot"]),
    "matplotlib": ("python", ["matplotlib"]),
    "pytorch": ("python", ["pytorch"]), "tensorflow": ("python", ["tensorflow"]),
    "express": ("javascript", ["express"]),
    "typescript": ("typescript", []), "ts": ("typescript", []),
}.items():
    _TAG_TO_META.setdefault(_alias, _meta)

# Question tags that are considered *technology* markers rather than languages.
_TECHNOLOGY_TAGS = {
    "react", "reactjs", "django", "flask", "pandas", "numpy", "mysql",
    "postgresql", "android", "ios", "docker", "git", "spring-boot", "angular",
    "vue.js", "laravel", "kubernetes", "node.js", "express", "matplotlib",
    "pytorch", "tensorflow", "spring", "rails", "ruby-on-rails", "bootstrap",
    "jquery", "webpack", "npm", "pip", "tensorflow2.0",
}
# Question tags treated as programming languages.
_LANGUAGE_TAGS = {
    "python", "javascript", "java", "c#", "c++", "typescript", "sql", "html",
    "css", "swift", "php", "ruby", "go", "r", "kotlin", "c", "objective-c",
    "scala", "rust", "dart", "perl", "bash", "powershell", "matlab",
}


def _pick_meta(tags: list[str]) -> tuple[str, list[str]]:
    """Derive (language, technologies) from a question's tag list."""
    language, techs = "", []
    for t in tags:
        if t in _LANGUAGE_TAGS and not language:
            language = t
        if t in _TECHNOLOGY_TAGS and t not in techs:
            techs.append(t)
    if not language and tags:
        # fall back to the alias table (e.g. tag 'reactjs' -> javascript)
        for t in tags:
            if t in _TAG_TO_META and _TAG_TO_META[t][0]:
                language = _TAG_TO_META[t][0]
                techs.extend(x for x in _TAG_TO_META[t][1] if x not in techs)
                break
    return language, techs[:10]


def _stable_doc_id(site: str, qid: int) -> str:
    return f"{site}-{qid}"


def extract_document(question: dict, answers: list[dict] | None) -> Document | None:
    """Map one raw question (+ its answers) to a Document, or None if unusable."""
    qid = question.get("question_id")
    if not qid:
        return None
    tags = [t.lower() for t in question.get("tags", [])]
    language, techs = _pick_meta(tags)

    title = html_to_text(question.get("title", ""))
    body_text = html_to_text(question.get("body", ""))
    code_blocks = extract_code_blocks(question.get("body", ""))

    # Choose the best answer: accepted answer wins, else highest score.
    best = None
    if answers:
        accepted = [a for a in answers if a.get("is_accepted")]
        pool = accepted or sorted(answers, key=lambda a: a.get("score", 0), reverse=True)
        best = pool[0] if pool else None
    if best is None:
        return None  # a "solution" document needs an answer

    answer_text = html_to_text(best.get("body", ""))
    answer_code = extract_code_blocks(best.get("body", ""))
    if len(title) < 10 or len(body_text) < 20 or len(answer_text) < 20:
        return None

    error_message = _extract_error_message(title, body_text)

    return Document(
        doc_id=_stable_doc_id("so", qid),
        title=title[:300],
        error_message=(error_message[:2000] if error_message else None),
        description=body_text[:6000],
        solution=answer_text[:8000],
        code_example=("\n\n".join(answer_code[:2])[:8000] if answer_code
                      else ("\n\n".join(code_blocks[:2])[:8000] if code_blocks else None)),
        tags=tags[:10],
        language=language or None,
        technologies=techs,
        source_url=question.get("link") or f"https://stackoverflow.com/questions/{qid}",
        source_site="stackoverflow",
        source_author=(question.get("owner") or {}).get("display_name"),
        source_author_url=(question.get("owner") or {}).get("link"),
        answer_author=(best.get("owner") or {}).get("display_name"),
        answer_author_url=(best.get("owner") or {}).get("link"),
        created_date=(question.get("creation_date")
                      and time.strftime("%Y-%m-%d", time.gmtime(question["creation_date"]))),
        score=float(best.get("score", 0) or 0),
        license=question.get("content_license") or "CC BY-SA 4.0",
        answer_license=best.get("content_license") or question.get("content_license") or "CC BY-SA 4.0",
    )


def _error_patterns() -> list[re.Pattern]:
    return [
        re.compile(r"(?:^|\n)\s*(?:Traceback \(most recent call last\)[\s\S]{0,600})", re.IGNORECASE),
        re.compile(r"\b(?:SyntaxError|IndentationError|NameError|TypeError|ValueError|"
                   r"AttributeError|KeyError|IndexError|ImportError|ModuleNotFoundError|"
                   r"RuntimeError|RecursionError|ZeroDivisionError|FileNotFoundError|"
                   r"PermissionError|ConnectionError|TimeoutError|OSError)\b[^\n]{0,300}"),
        re.compile(r"\b(?:NullPointerException|ArrayIndexOutOfBoundsException|"
                   r"ClassNotFoundException|IllegalArgumentException|"
                   r"NoClassDefFoundError|StackOverflowError)\b[^\n]{0,300}"),
        re.compile(r"\b(?:Segmentation fault|undefined reference|"
                   r"fatal error|cannot find|is not defined|is not a function|"
                   r"Unexpected token|Uncaught (?:TypeError|ReferenceError|SyntaxError))\b[^\n]{0,300}",
                   re.IGNORECASE),
        re.compile(r"\b(?:ERR_|Error:|ERROR:|FATAL|Exception in thread|"
                   r"Warning:|UnhandledPromiseRejection|ECONNREFUSED|ENOENT|EADDRINUSE)\s?[^\n]{0,300}"),
        re.compile(r"\b(?:HTTP|status code)\s*\d{3}\b[^\n]{0,200}", re.IGNORECASE),
        re.compile(r"\berror\b[^\n]{0,200}", re.IGNORECASE),
    ]


def _extract_error_message(title: str, body: str) -> str | None:
    """Best-effort extraction of the verbatim error text from title/body."""
    for pat in _error_patterns():
        m = pat.search(body or "")
        if m:
            line = m.group(0).strip().splitlines()
            return line[0][:500] if line else None
    m = re.search(r"\b(error|exception|undefined|cannot|failed|unexpected)\b[^\n]{0,200}",
                  title or "", re.IGNORECASE)
    return m.group(0)[:500] if m else None


# ----------------------------------------------------------------------
# Cleaning and duplicate detection (IDX-06)
# ----------------------------------------------------------------------
_WS_RE = re.compile(r"\s+")
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def _normalized_fingerprint(text: str) -> str:
    """Hash of whitespace/case-normalized text — catches exact dupes."""
    return hashlib.sha1(_WS_RE.sub(" ", text or "").strip().lower().encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 5) -> set[str]:
    words = _NONALNUM_RE.sub(" ", (text or "").lower()).split()
    return {" ".join(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(docs: list[Document], near_dup_threshold: float = 0.85) -> tuple[list[Document], dict]:
    """Remove exact duplicates (title+body hash) and near-duplicates
    (5-shingle Jaccard over title+description). Returns (kept, stats)."""
    kept: list[Document] = []
    seen_exact: set[str] = set()
    shingle_index: list[tuple[set[str], int]] = []  # (shingles, index in kept)
    exact_removed = near_removed = 0

    for d in docs:
        fp = _normalized_fingerprint(d.title + " " + d.description)
        if fp in seen_exact:
            exact_removed += 1
            continue
        sh = _shingles(d.title + " " + d.description)
        is_near = False
        for prev_sh, prev_idx in shingle_index:
            # Cheap pre-filter: shingle-set size ratio.
            if abs(len(prev_sh) - len(sh)) > 0.5 * max(len(prev_sh), len(sh)):
                continue
            if _jaccard(prev_sh, sh) >= near_dup_threshold:
                is_near = True
                break
        if is_near:
            near_removed += 1
            continue
        seen_exact.add(fp)
        shingle_index.append((sh, len(kept)))
        kept.append(d)

    stats = {"exact_duplicates_removed": exact_removed,
             "near_duplicates_removed": near_removed,
             "kept": len(kept)}
    return kept, stats


# ----------------------------------------------------------------------
# Pipeline: raw -> corpus
# ----------------------------------------------------------------------
def build_corpus(raw_path: Path | None = None, out_path: Path | None = None) -> dict:
    """Full extraction pipeline. Returns dataset statistics (IDX-07)."""
    raw_path = raw_path or (config.RAW_DIR / "stackexchange_raw.jsonl")
    out_path = out_path or config.CORPUS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Group raw questions with their answers. The crawler keeps questions and
    # answers in separate replayable files; accept a combined file too so the
    # extractor remains useful for hand-built fixtures and older crawls.
    answers_by_qid: dict[int, list[dict]] = {}
    questions: list[dict] = []
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "question_id" in rec and "body" in rec:
                    questions.append(rec)
                elif "question_id" in rec:
                    answers_by_qid.setdefault(rec["question_id"], []).append(rec)

    # Normal crawl output stores answer objects in a sibling file. Loading it
    # here is essential: a question without its accepted/top answer cannot be
    # presented as a developer solution.
    answers_path = raw_path.with_name("stackexchange_answers.jsonl")
    if answers_path.exists():
        with open(answers_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid is not None:
                    answers_by_qid.setdefault(qid, []).append(rec)

    candidates: list[Document] = []
    dropped = 0
    for q in questions:
        doc = extract_document(q, answers_by_qid.get(q.get("question_id"), []))
        if doc is None:
            dropped += 1
        else:
            candidates.append(doc)

    valid, errors = validate_documents(candidates)
    kept, dedup_stats = deduplicate(valid)

    with open(out_path, "w", encoding="utf-8") as f:
        for d in kept:
            f.write(d.model_dump_json() + "\n")

    stats = dataset_stats(kept, dropped, errors, dedup_stats)
    with open(config.STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


def dataset_stats(docs: list[Document], dropped: int, errors: list[dict],
                  dedup_stats: dict) -> dict:
    """Aggregate dataset statistics with full source traceability (IDX-07)."""
    lang_counts: dict[str, int] = {}
    tech_counts: dict[str, int] = {}
    with_error = with_code = 0
    for d in docs:
        if d.language:
            lang_counts[d.language] = lang_counts.get(d.language, 0) + 1
        for t in d.technologies:
            tech_counts[t] = tech_counts.get(t, 0) + 1
        if d.error_message:
            with_error += 1
        if d.code_example:
            with_code += 1
    return {
        "total_documents": len(docs),
        "dropped_no_answer_or_short": dropped,
        "schema_validation_errors": len(errors),
        "dedup": dedup_stats,
        "documents_with_error_message": with_error,
        "documents_with_code_example": with_code,
        "languages": dict(sorted(lang_counts.items(), key=lambda kv: -kv[1])),
        "technologies": dict(sorted(tech_counts.items(), key=lambda kv: -kv[1])),
        "distinct_languages": len(lang_counts),
        "distinct_technologies": len(tech_counts),
        "sources": {"stackoverflow.com": len(docs)},
        "license": "CC BY-SA 4.0 (Stack Exchange content, attributed via source_url)",
    }


if __name__ == "__main__":
    config.ensure_dirs()
    s = build_corpus()
    print(json.dumps(s, indent=2))
