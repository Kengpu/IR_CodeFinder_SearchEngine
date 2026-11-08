"""Build the persisted lexical index from the processed corpus."""
from __future__ import annotations

from app.indexing.inverted import build_and_save


def main() -> None:
    index = build_and_save()
    print(f"indexed {index.total_docs} docs, {index.vocabulary_size} terms")


if __name__ == "__main__":
    main()
