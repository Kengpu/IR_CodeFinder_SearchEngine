# Dataset collection and attribution

## Source and collection

The current corpus was collected from Stack Overflow via the documented Stack Exchange API (`https://api.stackexchange.com/2.3`). Queries covered the configured programming-language and technology tags, sorted by votes/activity. Requests include a User-Agent, obey the API's returned `backoff`, and run with a delay. The anonymous API began returning HTTP 429 near the end of the final expansion; the crawler retained collected responses and skipped failed tail batches.

The saved run yielded **2,725 normalized Q&A documents**, from **5,200 collected question bodies and 5,100 answer records**. Each normalized document uses the API's returned canonical question URL; author display names/profile links and `content_license` are retained when present. The solution is the accepted answer when available, otherwise the highest-scored fetched answer. Records without a usable question and answer are dropped.

## License and attribution

Stack Overflow content is distributed under the content license returned by the API (for this snapshot, typically CC BY-SA 4.0). Each record retains `license`, the question's `source_author` and profile URL, the selected answer's `answer_author` and profile URL, and the canonical question source URL. Details pages expose the source and author attribution. If redistributing content, preserve attribution and comply with the applicable CC BY-SA version's share-alike and attribution requirements. The dataset is a research/educational snapshot, not a substitute for reviewing Stack Overflow's current terms and license notices.

The API may omit owner information for deleted/unregistered accounts. In that case the source and license still remain traceable, but no author name is fabricated.

## Validation and statistics

`data/processed/stats.json` is generated from the corpus and records counts by language/technology, validation drops, duplicate counts, and the license note. `data/raw/` stores the raw API questions and answers so extraction can be replayed. To rebuild:

```bash
PYTHONPATH=backend python scripts/expand_dataset.py --max-questions 6000
PYTHONPATH=backend python -c "from app.crawler.extract import build_corpus; print(build_corpus())"
```

The crawler is rate-limited; HTTP 429 responses should be respected and retried later rather than bypassed.

## Known limitations

- Collection is biased toward highly voted/activity-sorted questions, English text, and configured tags.
- Error messages are extracted heuristically; only a subset contains a distinct `error_message` field, while the original problem remains in `description`.
- The corpus covers 19 languages and 28 technologies as inferred from source tags; inferred metadata is not manual annotation.
- The evaluation qrels are deterministic seed judgments anchored to corpus records, not judgments by independent human assessors.
- Stack Exchange API limits and content terms may change. Recheck current terms/licenses before a new crawl or redistribution.

## Provenance

- Source endpoint: `https://api.stackexchange.com/2.3`
- Site: Stack Overflow
- Raw data: `data/raw/stackexchange_raw.jsonl`, `data/raw/stackexchange_answers.jsonl`
- Processed data: `data/processed/corpus.jsonl`
- Statistics: `data/processed/stats.json`
- License field: per-document `license`
- Attribution fields: `source_author`, `source_author_url`, `answer_author`, `answer_author_url`
