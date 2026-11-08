from __future__ import annotations

import unittest

from app.ranking.fusion import (field_rerank, reciprocal_rank_fusion,
                                weighted_score_fusion)
from app.retrieval.lexical import SearchHit


def hit(doc_id: str, score: float) -> SearchHit:
    return SearchHit(doc_id, score, {"method": "test"})


class FusionTests(unittest.TestCase):
    def test_rrf_prefers_consistent_document(self):
        result = reciprocal_rank_fusion(
            [[hit("a", 10), hit("b", 9)], [hit("b", .9), hit("a", .8)]],
            k=1, top_k=2)
        self.assertEqual(result[0].doc_id, "a")  # equal ranks tie breaks by id
        self.assertEqual(set(h.doc_id for h in result), {"a", "b"})
        self.assertIn("source_ranks", result[0].explanation)

    def test_weighted_fusion_normalizes_different_scales(self):
        result = weighted_score_fusion(
            [hit("a", 100), hit("b", 50)], [hit("b", .99), hit("c", .1)],
            lexical_weight=.4, semantic_weight=.6, top_k=3)
        self.assertEqual(result[0].doc_id, "b")
        self.assertAlmostEqual(result[0].explanation["normalized_lexical_score"], 0.0)
        self.assertAlmostEqual(result[0].explanation["normalized_semantic_score"], 1.0)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            weighted_score_fusion([], [], lexical_weight=0, semantic_weight=0)

    def test_reranker_explains_fields(self):
        docs = {"a": {"title": "TypeError Python", "error_message": "TypeError"},
                "b": {"title": "Unrelated", "error_message": "Other"}}
        result = field_rerank([hit("b", 1), hit("a", 1)], docs, ["typeerror", "python"])
        self.assertEqual(result[0].doc_id, "a")
        self.assertIn("title_term_overlap", result[0].explanation)


if __name__ == "__main__":
    unittest.main()
