from __future__ import annotations

import unittest

from app.evaluation.graph import build_document_graph, hits, pagerank
from app.evaluation.metrics import (average_precision, evaluate_query_set,
                                     ndcg_at_k, recall_at_k)
from app.evaluation.dataset import (create_qrels, create_evaluation_queries,
                                     distinctive_error_terms, usable_error_text)
from app.models.document import Document


def make_doc(doc_id: str, title: str, error: str, language: str = "python") -> Document:
    return Document(doc_id=doc_id, title=title, error_message=error,
                    description="A sufficiently long problem description for testing.",
                    solution="A sufficiently long solution description for testing.",
                    tags=[language, "error"], language=language, technologies=["django"],
                    source_url=f"https://stackoverflow.com/questions/{doc_id.split('-')[-1]}",
                    source_site="stackoverflow")


class ErrorTextFilterTests(unittest.TestCase):
    def test_keeps_specific_diagnostics(self):
        for text in ("TypeError", "NullPointerException", "ENOENT",
                     "ValueError: setting an array element with a sequence",
                     "fatal error: Python.h: No such file or directory",
                     "cannot find module react"):
            self.assertEqual(usable_error_text(text), text, text)

    def test_rejects_bare_labels_and_fragments(self):
        # These real corpus values passed the old len(error) >= 5 guard and
        # became meaningless evaluation queries.
        for text in ("error", "error:", "error.", "error?", "ERROR", "exception",
                     "error message:", "Traceback (most recent call last)",
                     "", None, "   "):
            self.assertEqual(usable_error_text(text), "", repr(text))

    def test_rejects_prose_sentences_about_an_error(self):
        """`extract.py` falls back to a bare \\berror\\b pattern, so the corpus
        contains sentences that merely mention an error."""
        for text in ("error or warning.", "error when compiling in Swift.",
                     "error that says:", "error if that's missing).",
                     "error? If the compiler doesn't check the type"):
            self.assertEqual(usable_error_text(text), "", repr(text))

    def test_keeps_labelled_errors_carrying_an_identifier(self):
        for text in ("error: Untracked working tree file 'example.txt' would be lost",
                     "error reporting is `E_ALL`. I have restarted Apache.",
                     "Error: This is R 2.11.1, package 'snow' needs >= 2.12.1"):
            self.assertEqual(usable_error_text(text), text, repr(text))

    def test_strips_surrounding_quotes_and_whitespace(self):
        self.assertEqual(usable_error_text("  `TypeError: bad operand`  "),
                         "TypeError: bad operand")


class DistinctiveErrorTermTests(unittest.TestCase):
    def test_corpus_wide_terms_are_not_distinctive(self):
        # 'error' appears in every document; 'nullpointerexception' in one.
        docs = [make_doc(f"so-{n}", f"Title {n}", f"error: Oops {n} NullPointerError")
                for n in range(50)]
        docs.append(make_doc("so-99", "Rare", "NullPointerException: unique failure"))
        distinctive = distinctive_error_terms(docs)
        self.assertNotIn("error", distinctive)
        self.assertIn("nullpointerexception", distinctive)

    def test_empty_corpus_is_safe(self):
        self.assertEqual(distinctive_error_terms([]), set())


class QrelEvidenceTests(unittest.TestCase):
    def test_shared_generic_label_does_not_confer_relevance_two(self):
        """A bare 'error:' label must not mark unrelated documents relevant."""
        docs = [make_doc("so-1", "Python list problem", "error:"),
                make_doc("so-2", "Django template problem", "error:"),
                make_doc("so-3", "Java threading problem", "error:", "java")]
        queries = [{"query_id": "q001", "anchor_doc_id": "so-1"}]
        qrels = {row["doc_id"]: row["relevance"]
                 for row in create_qrels(queries, docs)}
        self.assertEqual(qrels["so-1"], 3)
        self.assertNotEqual(qrels.get("so-2"), 2)
        self.assertNotEqual(qrels.get("so-3"), 2)

    def test_shared_distinctive_error_confers_relevance_two(self):
        docs = [make_doc("so-1", "First", "ImportError: attempted relative import"),
                make_doc("so-2", "Second", "ImportError: attempted relative import"),
                make_doc("so-3", "Third", "ZeroDivisionError: division by zero")]
        # Pad with unrelated documents so the shared term stays rare.
        docs += [make_doc(f"so-pad{n}", f"Pad {n}", f"PadError{n}: filler text here")
                 for n in range(30)]
        queries = [{"query_id": "q001", "anchor_doc_id": "so-1"}]
        qrels = {row["doc_id"]: row["relevance"]
                 for row in create_qrels(queries, docs)}
        self.assertEqual(qrels["so-1"], 3)
        self.assertEqual(qrels["so-2"], 2)
        self.assertNotEqual(qrels.get("so-3"), 2)


class EvaluationQueryTests(unittest.TestCase):
    def test_query_types_cover_exact_and_paraphrase(self):
        """Query type must follow eligibility, not list position.

        Previously the paraphrase template keyed off a document's index in the
        full list, so a corpus whose eligible documents all landed elsewhere
        produced zero paraphrase queries.
        """
        docs = []
        for n in range(12):
            # Only every other document carries usable error text.
            error = f"CustomError{n}: specific failure number {n}" if n % 2 == 0 else "error:"
            docs.append(make_doc(f"so-{n:03d}", f"Problem number {n}", error,
                                 language=f"lang{n % 3}"))
        queries = create_evaluation_queries(docs, 9)
        types = {q["query_type"] for q in queries}
        self.assertIn("exact_error", types)
        self.assertIn("paraphrase", types)

    def test_queries_are_never_bare_labels(self):
        docs = [make_doc(f"so-{n}", f"Distinct problem title {n}", "error:")
                for n in range(6)]
        queries = create_evaluation_queries(docs, 5)
        for query in queries:
            self.assertNotIn(query["query"].strip().casefold(),
                             {"error", "error:", "error."})
            self.assertEqual(query["query_type"], "natural_question")

    def test_anchors_are_unique(self):
        docs = [make_doc(f"so-{n}", f"Title {n}", f"ErrorKind{n}: detail {n}")
                for n in range(10)]
        queries = create_evaluation_queries(docs, 10)
        anchors = [q["anchor_doc_id"] for q in queries]
        self.assertEqual(len(anchors), len(set(anchors)))


class MetricTests(unittest.TestCase):
    def test_metrics_known_values_and_edge_cases(self):
        ranked = ["a", "c", "b"]
        qrels = {"a": 3, "c": 1}
        self.assertAlmostEqual(ndcg_at_k(ranked, qrels, 3), 1.0)
        self.assertAlmostEqual(average_precision(ranked, qrels), 1.0)
        self.assertAlmostEqual(recall_at_k(ranked, qrels, 1), .5)
        self.assertEqual(ndcg_at_k([], {}, 10), 0.0)
        self.assertEqual(evaluate_query_set({"q": []}, {"q": {} }),
                         {"ndcg@5": 0.0, "recall@5": 0.0, "ndcg@10": 0.0,
                          "recall@10": 0.0, "map": 0.0})

    def test_query_set_macro_metrics(self):
        metrics = evaluate_query_set({"q1": ["a"], "q2": ["z"]},
                                     {"q1": {"a": 1}, "q2": {"b": 1}})
        self.assertEqual(metrics["judged_queries"], 2.0)
        self.assertAlmostEqual(metrics["recall@5"], .5)


class GraphTests(unittest.TestCase):
    def test_pagerank_and_hits_are_normalized_and_deterministic(self):
        graph = {"a": ["b"], "b": ["a"], "c": []}
        ranks = pagerank(graph)
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=8)
        authorities, hubs = hits(graph)
        self.assertGreaterEqual(authorities["a"], 0)
        self.assertGreaterEqual(hubs["b"], 0)

    def test_graph_from_documents_and_qrels_only_known_ids(self):
        docs = [make_doc("so-1", "Python error", "TypeError"),
                make_doc("so-2", "Python error again", "TypeError"),
                make_doc("so-3", "Java error", "NullPointerException", "java")]
        graph = build_document_graph(docs)
        self.assertEqual(set(graph), {"so-1", "so-2", "so-3"})
        queries = create_evaluation_queries(docs, 3)
        qrels = create_qrels(queries, docs)
        self.assertTrue(qrels)
        self.assertTrue(all(row["doc_id"] in graph for row in qrels))


if __name__ == "__main__":
    unittest.main()
