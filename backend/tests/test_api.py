from __future__ import annotations

import unittest

import faiss
import numpy as np
from fastapi.testclient import TestClient

from app.indexing.inverted import build_index
from app.models.document import Document
from app.retrieval.lexical import LexicalSearcher
from app.retrieval.semantic import SemanticSearcher
from app.services.search_pipeline import SearchPipeline
from app.main import create_app


class FakeModel:
    def encode(self, texts, **kwargs):
        output = []
        for text in texts:
            text = text.casefold()
            output.append([1.0, 0.0] if "python" in text or "typeerror" in text else [0.0, 1.0])
        vectors = np.asarray(output, dtype="float32")
        if kwargs.get("normalize_embeddings"):
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def fixture_pipeline() -> SearchPipeline:
    docs = [
        Document(doc_id="so-1001", title="Python TypeError unsupported operand",
                 error_message="TypeError: unsupported operand type(s)",
                 description="Python cannot add an integer and string value in this example.",
                 solution="Convert the string to an integer before adding the values.",
                 code_example="int(value)", tags=["python"], language="python",
                 technologies=["django"], source_url="https://stackoverflow.com/questions/1001",
                 source_site="stackoverflow"),
        Document(doc_id="so-1002", title="JavaScript TypeError is not a function",
                 error_message="TypeError: value is not a function",
                 description="JavaScript calls a value that is not a function.",
                 solution="Check the variable type and make sure the function is imported.",
                 code_example="typeof value", tags=["javascript"], language="javascript",
                 technologies=["react"], source_url="https://stackoverflow.com/questions/1002",
                 source_site="stackoverflow"),
    ]
    index = build_index(docs)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    vector_index = faiss.IndexFlatIP(2)
    vector_index.add(vectors)
    semantic = SemanticSearcher(vector_index, [d.doc_id for d in docs], index.documents,
                                FakeModel(), model_name="api-test")
    return SearchPipeline(LexicalSearcher(index), semantic)


class ApiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app(fixture_pipeline()))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["lexical_index"])
        self.assertTrue(response.json()["semantic_index"])
        self.assertEqual(response.json()["documents"], 2)

    def test_every_search_mode_uses_backend(self):
        for mode in ("tfidf", "bm25", "boolean", "semantic", "hybrid_rrf",
                     "hybrid_weighted", "hybrid_prf"):
            response = self.client.get("/search", params={"q": "Python TypeError", "mode": mode})
            self.assertEqual(response.status_code, 200, (mode, response.text))
            payload = response.json()
            self.assertEqual(payload["mode"], mode)
            self.assertGreater(payload["total"], 0)
            self.assertEqual(payload["results"][0]["doc_id"], "so-1001")
            self.assertIn("ranking_explanation", payload["results"][0])
            self.assertIn("<mark>", payload["results"][0]["highlighted_snippet"])

    def test_filters_and_empty_results(self):
        response = self.client.get("/search", params={"q": "TypeError", "mode": "bm25",
                                                        "language": "javascript"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([r["doc_id"] for r in response.json()["results"]], ["so-1002"])
        self.assertEqual(self.client.get("/search", params={"q": "nothingmatches", "mode": "bm25"}).json()["total"], 0)

    def test_invalid_and_blank_queries(self):
        self.assertEqual(self.client.get("/search", params={"q": "   "}).status_code, 422)
        self.assertEqual(self.client.get("/search", params={"q": "error", "mode": "wat"}).status_code, 422)
        self.assertEqual(self.client.get("/search", params={"q": "error", "top_k": 1000}).status_code, 422)

    def test_document_and_similar_endpoints(self):
        detail = self.client.get("/documents/so-1001")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["doc_id"], "so-1001")
        self.assertIn("related_errors", detail.json())
        similar = self.client.get("/similar/so-1001")
        self.assertEqual(similar.status_code, 200)
        self.assertNotIn("so-1001", [x["doc_id"] for x in similar.json()["results"]])
        self.assertEqual(self.client.get("/documents/absent").status_code, 404)
        self.assertEqual(self.client.get("/similar/absent").status_code, 404)

    def test_filters_and_evaluation_endpoint(self):
        self.assertEqual(self.client.get("/languages").status_code, 200)
        self.assertEqual(self.client.get("/technologies").status_code, 200)
        evaluation = self.client.get("/evaluation")
        self.assertEqual(evaluation.status_code, 200)
        payload = evaluation.json()
        self.assertIn("available", payload)
        self.assertIn("metrics", payload)
        if payload["available"]:
            self.assertIn("metrics", payload["metrics"])
        else:
            self.assertIsInstance(payload["message"], str)


if __name__ == "__main__":
    unittest.main()
