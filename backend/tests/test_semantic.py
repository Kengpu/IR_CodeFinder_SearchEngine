from __future__ import annotations

import unittest

import faiss
import numpy as np

from app.retrieval.semantic import SemanticSearcher


class FakeModel:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = text.casefold()
            if "python" in value or "typeerror" in value:
                vectors.append([1.0, 0.0, 0.0])
            elif "javascript" in value:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        array = np.asarray(vectors, dtype="float32")
        if kwargs.get("normalize_embeddings"):
            array /= np.linalg.norm(array, axis=1, keepdims=True)
        return array


class SemanticSearcherTests(unittest.TestCase):
    def setUp(self):
        self.documents = {
            "so-1": {"doc_id": "so-1", "title": "Python TypeError", "error_message": "TypeError",
                     "description": "Python cannot add a string", "language": "python",
                     "technologies": ["django"]},
            "so-2": {"doc_id": "so-2", "title": "JavaScript TypeError", "error_message": "TypeError",
                     "description": "JavaScript function call", "language": "javascript",
                     "technologies": ["react"]},
            "so-3": {"doc_id": "so-3", "title": "Python file error", "error_message": "FileNotFoundError",
                     "description": "Python missing file", "language": "python",
                     "technologies": ["flask"]},
        }
        vectors = np.asarray([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]], dtype="float32")
        self.faiss_index = faiss.IndexFlatIP(3)
        self.faiss_index.add(vectors)
        self.searcher = SemanticSearcher(self.faiss_index, list(self.documents), self.documents,
                                         FakeModel(), model_name="fake-test-model")

    def test_semantic_ranks_by_cosine(self):
        hits = self.searcher.search("Python TypeError", top_k=2)
        self.assertEqual(hits[0].doc_id, "so-1")
        self.assertEqual(hits[1].doc_id, "so-3")
        self.assertAlmostEqual(hits[0].score, 1.0)

    def test_metadata_filter(self):
        hits = self.searcher.search("TypeError", language="javascript")
        self.assertEqual([h.doc_id for h in hits], ["so-2"])
        self.assertEqual(self.searcher.search("TypeError", technology="none"), [])

    def test_similar_excludes_self(self):
        hits = self.searcher.similar("so-1", top_k=2)
        self.assertNotIn("so-1", [h.doc_id for h in hits])
        self.assertEqual(hits[0].doc_id, "so-3")

    def test_id_alignment_validation(self):
        with self.assertRaises(ValueError):
            SemanticSearcher(self.faiss_index, ["so-1"], self.documents, FakeModel())

    def test_empty_query(self):
        self.assertEqual(self.searcher.search("  "), [])


if __name__ == "__main__":
    unittest.main()
