from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.indexing.inverted import InvertedIndex, build_index
from app.indexing.tokenizer import tokenize
from app.models.document import Document
from app.retrieval.lexical import LexicalSearcher


def doc(doc_id: str, title: str, description: str, solution: str,
        language: str = "python", technologies: list[str] | None = None) -> Document:
    return Document(
        doc_id=doc_id, title=title, description=description, solution=solution,
        error_message=title, code_example="print('ok')", tags=[language],
        language=language, technologies=technologies or [],
        source_url=f"https://stackoverflow.com/questions/{doc_id.split('-')[-1]}",
        source_site="stackoverflow",
    )


class TokenizerTests(unittest.TestCase):
    def test_preserves_programming_tokens(self):
        tokens = tokenize(r"TypeError 404 == => :: () {} C:\\path /usr/local/bin npm install pip install",
                          remove_stopwords=False)
        for expected in ["TypeError", "404", "==", "=>", "::", "(", ")", "{", "}",
                         r"C:\\path", "/usr/local/bin", "npm_install", "pip_install"]:
            self.assertIn(expected, tokens)

    def test_stopwords_remove_prose_only(self):
        tokens = tokenize("the error is TypeError in Python", remove_stopwords=True)
        self.assertNotIn("the", tokens)
        self.assertIn("TypeError", tokens)
        self.assertIn("python", tokens)

    def test_empty_input(self):
        self.assertEqual(tokenize(""), [])


class IndexAndRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            doc("so-101", "Python TypeError: unsupported operand", "Add two values in Python",
                "Convert the string to an integer before adding.", technologies=["python"]),
            doc("so-202", "JavaScript TypeError is not a function", "Calling a value in JavaScript",
                "Check that the variable references a function.", "javascript", ["react"]),
            doc("so-303", "Python file not found error", "Opening a missing file in Python",
                "Use the correct absolute path.", technologies=["django"]),
        ]
        self.index = build_index(self.docs, remove_stopwords=True)
        self.searcher = LexicalSearcher(self.index)

    def test_index_tf_df_and_metadata(self):
        self.assertEqual(self.index.total_docs, 3)
        self.assertGreater(self.index.vocabulary_size, 0)
        self.assertIn("python", self.index.postings)
        self.assertEqual(len(self.index.postings["python"]), 2)
        self.assertEqual(self.index.filtered_doc_ids(language="javascript"), {"so-202"})
        self.assertEqual(self.index.filtered_doc_ids(technology="react"), {"so-202"})

    def test_tfidf_and_bm25_rank_relevant(self):
        self.assertEqual(self.searcher.tfidf("Python TypeError")[0].doc_id, "so-101")
        self.assertEqual(self.searcher.bm25("Python TypeError")[0].doc_id, "so-101")

    def test_boolean_and_or_not(self):
        self.assertEqual({h.doc_id for h in self.searcher.boolean("Python AND TypeError")},
                         {"so-101"})
        self.assertEqual({h.doc_id for h in self.searcher.boolean("Python OR JavaScript")},
                         {"so-101", "so-202", "so-303"})
        self.assertEqual({h.doc_id for h in self.searcher.boolean("Python AND NOT TypeError")},
                         {"so-303"})

    def test_filters(self):
        hits = self.searcher.bm25("TypeError", language="javascript")
        self.assertEqual([h.doc_id for h in hits], ["so-202"])
        self.assertEqual(self.searcher.bm25("TypeError", technology="missing"), [])

    def test_bm25_matches_independent_reference(self):
        for query in ("Python TypeError", "missing term", "file not found"):
            actual = self.searcher.bm25(query, top_k=10)
            reference = self.searcher.reference_bm25(query, top_k=10)
            self.assertEqual([h.doc_id for h in actual], [h.doc_id for h in reference])
            for a, b in zip(actual, reference):
                self.assertAlmostEqual(a.score, b.score, places=10)

    def test_index_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.index.save(root / "idx.pkl", root / "lengths.pkl", root / "meta.json",
                            root / "metadata.db")
            loaded = InvertedIndex.load(root / "idx.pkl", root / "lengths.pkl", root / "meta.json")
            self.assertEqual(loaded.total_docs, self.index.total_docs)
            self.assertEqual(loaded.docs_for_term("TypeError"), self.index.docs_for_term("TypeError"))
            self.assertEqual(loaded.documents.keys(), self.index.documents.keys())

    def test_empty_and_unknown_queries(self):
        self.assertEqual(self.searcher.bm25(""), [])
        self.assertEqual(self.searcher.tfidf("unseenword"), [])
        self.assertEqual(self.searcher.boolean(""), [])


if __name__ == "__main__":
    unittest.main()
