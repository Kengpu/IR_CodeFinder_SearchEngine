"""Expand the Stack Exchange corpus with real, traceable Q&A records.

This script deliberately keeps collection separate from extraction:
1. collect question bodies and answer bodies through the public API;
2. write raw JSONL artifacts;
3. run the shared extractor to validate/deduplicate documents.

Usage (from project root):
    PYTHONPATH=backend python scripts/expand_dataset.py --max-questions 650
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from app import config
from app.crawler.stackexchange import TAG_QUERIES


class Api:
    def __init__(self, delay: float = 0.65, budget: int = 240):
        self.delay = delay
        self.budget = budget
        self.calls = 0
        self.backoff = 0

    def get(self, path: str, params: dict) -> dict:
        if self.calls >= self.budget:
            raise RuntimeError("Stack Exchange API request budget exhausted")
        if self.backoff:
            time.sleep(self.backoff)
            self.backoff = 0
        time.sleep(self.delay)
        query = dict(params)
        query["site"] = config.STACKEXCHANGE_SITE
        query["filter"] = "withbody"
        url = f"{config.STACKEXCHANGE_API}/{path}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "CodeFinder-IR-project/1.0 (educational dataset)"}
        )
        with urllib.request.urlopen(req, timeout=35) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.calls += 1
        if payload.get("backoff"):
            self.backoff = int(payload["backoff"])
        if payload.get("error_message"):
            raise RuntimeError(payload["error_message"])
        return payload


def collect_questions(api: Api, target: int) -> dict[int, dict]:
    """Collect across every configured technology using two independent sorts."""
    questions: dict[int, dict] = {}
    for tag, _language, _technologies in TAG_QUERIES:
        for sort in ("votes", "activity"):
            try:
                payload = api.get("questions", {
                    "tagged": tag, "pagesize": 100, "page": 1,
                    "order": "desc", "sort": sort,
                })
            except Exception as exc:
                print(f"[warn] questions {tag}/{sort}: {exc}")
                continue
            added = 0
            for question in payload.get("items", []):
                qid = question.get("question_id")
                # Keep records that have an answer; extraction will select the
                # accepted/highest-scored answer after the answer fetch.
                if qid and qid not in questions:
                    questions[qid] = question
                    added += 1
            print(f"{tag:16s} {sort:8s} +{added:3d} total={len(questions)}")
            if len(questions) >= target:
                return dict(list(questions.items())[:target])
    return dict(list(questions.items())[:target])


def fetch_bodies_and_answers(api: Api, questions: dict[int, dict]) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    ids = list(questions)
    bodies: dict[int, dict] = {}
    answers: dict[int, list[dict]] = {}
    for start in range(0, len(ids), 100):
        chunk = ids[start:start + 100]
        try:
            payload = api.get(f"questions/{';'.join(map(str, chunk))}", {
                "pagesize": 100, "order": "desc", "sort": "votes",
            })
            for item in payload.get("items", []):
                bodies[item["question_id"]] = item
        except Exception as exc:
            print(f"[warn] question bodies batch {start}: {exc}")
        print(f"bodies {len(bodies)}/{len(ids)}")

        try:
            payload = api.get(f"questions/{';'.join(map(str, chunk))}/answers", {
                "pagesize": 100, "order": "desc", "sort": "votes",
            })
            for answer in payload.get("items", []):
                answers.setdefault(answer["question_id"], []).append(answer)
        except Exception as exc:
            print(f"[warn] answers batch {start}: {exc}")
        print(f"answers for {min(start + 100, len(ids))}/{len(ids)} questions")
    return bodies, answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-questions", type=int, default=650)
    parser.add_argument("--delay", type=float, default=0.65)
    args = parser.parse_args()
    config.ensure_dirs()
    api = Api(delay=args.delay)
    questions = collect_questions(api, args.max_questions)
    bodies, answers = fetch_bodies_and_answers(api, questions)
    raw_questions = config.RAW_DIR / "stackexchange_raw.jsonl"
    raw_answers = config.RAW_DIR / "stackexchange_answers.jsonl"
    with raw_questions.open("w", encoding="utf-8") as stream:
        for question in bodies.values():
            stream.write(json.dumps(question, ensure_ascii=False) + "\n")
    with raw_answers.open("w", encoding="utf-8") as stream:
        for answer_list in answers.values():
            for answer in answer_list:
                stream.write(json.dumps(answer, ensure_ascii=False) + "\n")
    print(f"Collected {len(bodies)} questions and {sum(map(len, answers.values()))} answers "
          f"using {api.calls} API requests")


if __name__ == "__main__":
    main()
