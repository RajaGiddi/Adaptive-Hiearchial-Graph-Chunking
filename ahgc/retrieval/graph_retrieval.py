"""Graph-aware retrieval utilities for AHGC.

This module demonstrates the value of the Adaptive Hierarchical Graph Chunking
graph during retrieval: given top-k chunk hits from a vector store, we can pull
in structured neighbors (the parent section, the next chunk) to provide the LLM
with more coherent local context.

If the graph backend is a networkx.DiGraph, we expand hits with graph neighbors.
If a simple dict fallback is supplied (from the AHGC graph builder fallback), we
skip expansion and return only the original hit chunks.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:  # optional: only needed for type hints and duck-typing
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - only executed when networkx missing
    nx = None  # type: ignore

from ahgc.indexing.vectorstore import VectorStore

__all__ = ["retrieve_with_graph"]


def _is_dict_graph(graph: Any) -> bool:
    """Return True if graph is the dict fallback structure."""
    return isinstance(graph, dict) and "nodes" in graph and "edges" in graph


def retrieve_with_graph(
    query: str,
    vectorstore: VectorStore,
    graph: Any,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve chunk contexts using AHGC graph expansion.

    Steps
    -----
    1. Search the vector store for top-k chunk hits for the query.
    2. For each hit, read its metadata to get `chunk_id`, `section_id`,
       `section_title`, `doc_id` (if available).
    3. In the graph (if networkx), find:
       - the chunk node itself,
       - its parent section node (incoming edge relation='has_chunk'),
       - its immediate "next" chunk node (edge relation='next') if present.
    4. Collect these into a context list, deduplicating by chunk_id.
    5. Return a list of dicts with keys: ``chunk_id``, ``section_title``,
       ``text``, ``score`` (carried from the original hit).

    Notes
    -----
    - If ``graph`` is the dict fallback, we skip expansion and include only the
      original hit chunks, still carrying the score forward.
    - The function returns chunk-level entries; the parent section is used to
      supply a consistent ``section_title``.
    """
    hits = vectorstore.search(query, k=k)
    contexts: List[Dict[str, Any]] = []
    seen: set[str] = set()

    # Decide whether we can expand via graph traversal
    dict_fallback = _is_dict_graph(graph)

    for hit in hits:
        chunk_id: str = str(hit.get("id") or hit.get("chunk_id") or "")
        if not chunk_id:
            # Defensive: skip malformed entries
            continue
        md: Dict[str, Any] = hit.get("metadata", {}) or {}
        section_title_md: str = str(md.get("section_title", ""))
        text_hit: str = str(hit.get("text", ""))
        score: float = float(hit.get("score", 0.0))

        # Default section title from metadata; may be overridden from graph
        section_title = section_title_md

        # Add the hit itself
        if chunk_id not in seen:
            contexts.append(
                {
                    "chunk_id": chunk_id,
                    "section_title": section_title,
                    "text": text_hit,
                    "score": score,
                }
            )
            seen.add(chunk_id)

        # If graph is a dict fallback, skip neighbor expansion
        if dict_fallback:
            continue

        # networkx expansion (duck-typed: in_edges/out_edges/nodes access)
        try:
            # Find parent section via incoming 'has_chunk' edge
            parent_section_id = None
            for u, v, data in graph.in_edges(chunk_id, data=True):  # type: ignore[attr-defined]
                if data.get("relation") == "has_chunk":
                    parent_section_id = u
                    break

            if parent_section_id is not None:
                # Prefer section title from graph if present
                try:
                    title_from_graph = graph.nodes[parent_section_id].get("title")  # type: ignore[attr-defined]
                    if isinstance(title_from_graph, str) and title_from_graph:
                        section_title = title_from_graph
                except Exception:
                    pass

            # Find immediate next chunk via outgoing 'next' edge
            next_chunk_id = None
            for u, v, data in graph.out_edges(chunk_id, data=True):  # type: ignore[attr-defined]
                if data.get("relation") == "next":
                    next_chunk_id = v
                    break

            if next_chunk_id and next_chunk_id not in seen:
                # Get next chunk text from graph node attrs if available
                next_text = ""
                try:
                    next_text_val = graph.nodes[next_chunk_id].get("text")  # type: ignore[attr-defined]
                    if isinstance(next_text_val, str):
                        next_text = next_text_val
                except Exception:
                    pass
                contexts.append(
                    {
                        "chunk_id": str(next_chunk_id),
                        "section_title": section_title,
                        "text": next_text,
                        "score": score,  # carry original hit score forward
                    }
                )
                seen.add(str(next_chunk_id))
        except Exception:
            # Be forgiving: if traversal fails for any reason, just proceed
            # with what we already added for this hit.
            continue

    return contexts
