"""Stack Exchange API crawler (IDX-03).

Collects real question/answer pairs about programming errors from Stack
Overflow via the official Stack Exchange API (https://api.stackexchange.com).

Compliance notes (IDX-01):
- The API is public, documented, and intended for reading site content.
  It enforces its own throttling (300 requests/day anonymous); we also add
  a per-request politeness delay and a bounded request budget.
- Content is licensed CC BY-SA; the license is recorded on every document.
- Every stored document keeps the canonical question URL (traceability,
  IDX-07). No URLs are synthesized.

Raw API responses are written to data/raw/ as JSONL so collection can be
audited and re-processed without re-crawling.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from .. import config

# Tag queries used to cover a broad set of languages/technologies (>=14).
# Each entry: (stackexchange tag, language label, [technology labels])
TAG_QUERIES: list[tuple[str, str, list[str]]] = [
    ("python", "python", []),
    ("javascript", "javascript", []),
    ("java", "java", []),
    ("c#", "c#", []),
    ("c++", "c++", []),
    ("typescript", "typescript", []),
    ("reactjs", "javascript", ["react"]),
    ("node.js", "javascript", ["node.js"]),
    ("django", "python", ["django"]),
    ("flask", "python", ["flask"]),
    ("pandas", "python", ["pandas"]),
    ("numpy", "python", ["numpy"]),
    ("sql", "sql", []),
    ("mysql", "sql", ["mysql"]),
    ("postgresql", "sql", ["postgresql"]),
    ("html", "html", []),
    ("css", "css", []),
    ("android", "java", ["android"]),
    ("swift", "swift", ["ios"]),
    ("php", "php", []),
    ("ruby-on-rails", "ruby", ["ruby-on-rails"]),
    ("go", "go", []),
    ("r", "r", []),
    ("docker", [], ["docker"]),
    ("git", [], ["git"]),
    ("spring-boot", "java", ["spring-boot"]),
    ("angular", "typescript", ["angular"]),
    ("vue.js", "javascript", ["vue.js"]),
    ("laravel", "php", ["laravel"]),
    ("kubernetes", [], ["kubernetes"]),
]

# Error-oriented search phrases. Combined with tags via the API's `intitle`
# or body search to bias collection toward *errors and solutions*.
ERROR_PHRASES = [
    "error",
    "exception",
    "undefined",
    "cannot",
    "failed",
    "not working",
    "unexpected",
    "invalid",
]

REQUEST_BUDGET = 400  # hard cap on API calls per crawl run


class StackExchangeCrawler:
    """Minimal, polite Stack Exchange API client focused on error Q&A."""

    def __init__(self, raw_dir: Path | None = None, delay: float | None = None):
        self.api = config.STACKEXCHANGE_API
        self.site = config.STACKEXCHANGE_SITE
        self.raw_dir = raw_dir or config.RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.delay = config.CRAWLER_DELAY_SECONDS if delay is None else delay
        self.requests_made = 0
        self.backoff_seconds = 0  # API 'backoff' field must be honored

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _get(self, path: str, params: dict) -> dict:
        """Perform one GET against the API, honoring backoff and budget."""
        if self.requests_made >= REQUEST_BUDGET:
            raise RuntimeError("Request budget exhausted; crawl run stopped.")
        if self.backoff_seconds:
            time.sleep(self.backoff_seconds)
            self.backoff_seconds = 0
        time.sleep(self.delay)
        params = dict(params)
        params["site"] = self.site
        params["filter"] = "withbody"
        url = f"{self.api}/{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "CodeFinder-IR-project/1.0 (educational)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.requests_made += 1
            data = json.loads(resp.read().decode("utf-8"))
        # The API inserts a mandatory wait via `backoff` — must be respected.
        if isinstance(data, dict) and data.get("backoff"):
            self.backoff_seconds = int(data["backoff"])
        if isinstance(data, dict) and data.get("error_message"):
            raise RuntimeError(f"API error: {data['error_message']}")
        return data

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    def search_questions(self, tag: str, phrase: str, pages: int = 1) -> list[dict]:
        """Search for questions matching a tag + error phrase, sorted by votes."""
        out: list[dict] = []
        for page in range(1, pages + 1):
            data = self._get("search/advanced", {
                "tagged": tag,
                "intitle": phrase,
                "pagesize": 100,
                "page": page,
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
            })
            out.extend(data.get("items", []))
            if not data.get("has_more"):
                break
        return out

    def fetch_answers(self, question_ids: list[int]) -> dict[int, list[dict]]:
        """Fetch answers for a batch of questions (up to 100 ids per call)."""
        answers: dict[int, list[dict]] = defaultdict(list)
        for i in range(0, len(question_ids), 100):
            chunk = question_ids[i:i + 100]
            data = self._get("questions/{ids}/answers".format(
                ids=";".join(str(q) for q in chunk)), {
                "pagesize": 100,
                "order": "desc",
                "sort": "votes",
                "filter": "withbody",
            })
            for a in data.get("items", []):
                answers.setdefault(a["question_id"], []).append(a)
        return answers

    # ------------------------------------------------------------------
    # Raw persistence
    # ------------------------------------------------------------------
    def crawl(self, per_tag_questions: int = 14) -> list[dict]:
        """Crawl all tag queries; write raw pages; return raw question dicts.

        Deduplicates by question id across tag queries before returning.
        """
        raw_path = self.raw_dir / "stackexchange_raw.jsonl"
        seen: dict[int, dict] = {}
        for tag, _lang, _tech in TAG_QUERIES:
            for phrase in ERROR_PHRASES[:4]:
                try:
                    qs = self.search_questions(tag, phrase, pages=1)
                except RuntimeError:
                    # Budget exhausted or API error: keep what we have.
                    break
                except Exception as e:  # network hiccup — skip this query
                    print(f"  [crawl] {tag}/{phrase}: {e}")
                    continue
                for q in qs:
                    qid = q.get("question_id")
                    if qid and qid not in seen:
                        seen[qid] = q
                if len(seen) >= per_tag_questions * len(TAG_QUERIES) * 4:
                    break
            with open(raw_path, "a", encoding="utf-8") as f:
                for q in seen.values():
                    f.write(json.dumps(q, ensure_ascii=False) + "\n")
        # Overwrite with the deduplicated set (file may contain dupes from
        # earlier partial runs).
        with open(raw_path, "w", encoding="utf-8") as f:
            for q in seen.values():
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"[crawl] collected {len(seen)} unique questions "
              f"({self.requests_made} API requests)")
        return list(seen.values())


def main() -> None:
    config.ensure_dirs()
    crawler = StackExchangeCrawler()
    crawler.crawl()


if __name__ == "__main__":
    main()
