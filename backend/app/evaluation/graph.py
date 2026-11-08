"""Link graph analysis: PageRank and HITS (EVL-09..11).

The corpus contains canonical source links but not a dense inter-document
link graph. This module constructs a transparent graph from shared tags,
language, and exact diagnostic terms, then computes PageRank/HITS from
scratch. Those scores may be used as bounded tie-break features in ranking;
the default search pipeline does not silently inject graph scores.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .. import config
from ..indexing.tokenizer import canonical_token, tokenize
from ..models.document import Document


def build_document_graph(docs: list[Document], *, max_neighbors: int = 20,
                         min_shared: int = 2) -> dict[str, list[str]]:
    """Connect documents sharing multiple tags or diagnostic terms."""
    inverted: dict[str, set[str]] = defaultdict(set)
    doc_features: dict[str, set[str]] = {}
    for doc in docs:
        features = {f"tag:{tag.casefold()}" for tag in doc.tags}
        features.update(f"tech:{tech.casefold()}" for tech in doc.technologies)
        if doc.language:
            features.add(f"lang:{doc.language.casefold()}")
        error_terms = tokenize(doc.error_message or "", remove_stopwords=False)
        features.update(f"err:{canonical_token(term)}" for term in error_terms
                        if len(term) >= 3 and any(ch.isalpha() for ch in term))
        doc_features[doc.doc_id] = features
        for feature in features:
            inverted[feature].add(doc.doc_id)
    graph: dict[str, list[str]] = {}
    for doc in docs:
        counts: defaultdict[str, int] = defaultdict(int)
        for feature in doc_features[doc.doc_id]:
            for candidate in inverted[feature]:
                if candidate != doc.doc_id:
                    counts[candidate] += 1
        neighbors = [doc_id for doc_id, overlap in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])) if overlap >= min_shared]
        graph[doc.doc_id] = neighbors[:max_neighbors]
    return graph


def pagerank(graph: dict[str, list[str]], *, damping: float = 0.85,
             tolerance: float = 1e-10, max_iterations: int = 200) -> dict[str, float]:
    """Iterative PageRank with dangling-node redistribution."""
    if not 0 < damping < 1:
        raise ValueError("damping must be between 0 and 1")
    nodes = list(graph)
    n = len(nodes)
    if not n:
        return {}
    ranks = {node: 1.0 / n for node in nodes}
    for _ in range(max_iterations):
        dangling = sum(ranks[node] for node in nodes if not graph.get(node))
        base = (1.0 - damping) / n + damping * dangling / n
        updated = {node: base for node in nodes}
        for source, targets in graph.items():
            if not targets:
                continue
            contribution = damping * ranks[source] / len(targets)
            for target in targets:
                if target in updated:
                    updated[target] += contribution
        delta = sum(abs(updated[node] - ranks[node]) for node in nodes)
        ranks = updated
        if delta < tolerance:
            break
    return ranks


def hits(graph: dict[str, list[str]], *, tolerance: float = 1e-10,
         max_iterations: int = 200) -> tuple[dict[str, float], dict[str, float]]:
    """Compute normalized HITS authority and hub scores."""
    nodes = list(graph)
    if not nodes:
        return {}, {}
    authorities = {node: 1.0 for node in nodes}
    hubs = {node: 1.0 for node in nodes}
    reverse: dict[str, list[str]] = defaultdict(list)
    for source, targets in graph.items():
        for target in targets:
            if target in authorities:
                reverse[target].append(source)
    for _ in range(max_iterations):
        new_auth = {node: sum(hubs[src] for src in reverse[node]) for node in nodes}
        norm = math.sqrt(sum(value * value for value in new_auth.values())) or 1.0
        new_auth = {node: value / norm for node, value in new_auth.items()}
        new_hubs = {node: sum(new_auth[target] for target in graph.get(node, [])
                              if target in new_auth) for node in nodes}
        norm = math.sqrt(sum(value * value for value in new_hubs.values())) or 1.0
        new_hubs = {node: value / norm for node, value in new_hubs.items()}
        delta = sum(abs(new_auth[node] - authorities[node]) +
                    abs(new_hubs[node] - hubs[node]) for node in nodes)
        authorities, hubs = new_auth, new_hubs
        if delta < tolerance:
            break
    return authorities, hubs


def save_graph(graph: dict[str, list[str]], path: Path | None = None) -> None:
    path = path or config.LINK_GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(graph, stream, indent=2)


def graph_report(docs: list[Document], path: Path | None = None) -> dict:
    """Build graph, compute PageRank/HITS, and persist scores for analysis."""
    graph = build_document_graph(docs)
    page = pagerank(graph)
    authority, hub = hits(graph)
    save_graph(graph, path or config.LINK_GRAPH_PATH)
    report = {
        "nodes": len(graph),
        "edges": sum(len(targets) for targets in graph.values()),
        "pagerank_top10": sorted(page.items(), key=lambda pair: -pair[1])[:10],
        "authority_top10": sorted(authority.items(), key=lambda pair: -pair[1])[:10],
        "hub_top10": sorted(hub.items(), key=lambda pair: -pair[1])[:10],
        "ranking_note": ("PageRank/HITS are available as optional, bounded tie-break "
                         "features; not applied by default to avoid graph-signal bias."),
    }
    score_path = config.PROCESSED_DIR / "graph_scores.json"
    with score_path.open("w", encoding="utf-8") as stream:
        json.dump({"pagerank": page, "authority": authority, "hub": hub}, stream)
    return report


__all__ = ["build_document_graph", "pagerank", "hits", "save_graph", "graph_report"]
