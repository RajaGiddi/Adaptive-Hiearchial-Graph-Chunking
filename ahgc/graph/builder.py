"""Graph builder for Adaptive Hierarchical Graph Chunking (AHGC).

This module builds a hierarchical representation linking a document to its
sections and their chunks. The resulting graph enables structured context
expansion during retrieval (e.g., navigating from a chunk up to its section or
across sequential chunks).

Preferred backend: ``networkx`` directed graph (``networkx.DiGraph``).
Fallback: If ``networkx`` is unavailable, a simple Python dictionary structure
is returned with the following shape::

	{
		"nodes": {
			node_id: {"type": ..., "title": ..., "text": ..., ...},
			...
		},
		"edges": [ (src_id, dst_id, {"relation": ...}), ... ]
	}

The fallback preserves all attributes and relations but does not provide any of
the traversal utilities found in ``networkx``. Callers can treat it uniformly
by always accessing ``graph.nodes`` / ``graph.edges`` where available and
otherwise using the dict directly.

Configuration
-------------
Graph construction can be configured via ``GraphBuildConfig`` to enable or
disable certain structural edges. By default, AHGC adds sequential ``next``
edges between chunks in the same section. Additional shapes like sibling or
cross-section edges can be toggled for experimentation and benchmarking.
"""

from __future__ import annotations

from typing import Any, Dict, List
from dataclasses import dataclass

try:  # Optional dependency
	import networkx as nx  # type: ignore
	_HAS_NX = True
except Exception:  # pragma: no cover - only executed when networkx missing
	nx = None  # type: ignore
	_HAS_NX = False

__all__ = ["GraphBuildConfig", "build_graph_from_sections"]


@dataclass
class GraphBuildConfig:
	"""Configuration for graph construction in AHGC.

	Attributes
	----------
	add_next_edges : bool
		Whether to add sequential ``next`` edges between chunks within the same
		section. Default: True.
	add_sibling_edges : bool
		Placeholder toggle for adding edges between chunk siblings (e.g.,
		bidirectional edges among chunks within the same section). Not yet
		implemented. Default: False.
	add_cross_section_edges : bool
		Placeholder toggle for adding edges that connect chunks across sections
		(e.g., thematically related links). Not yet implemented. Default: False.
	"""

	add_next_edges: bool = True
	add_sibling_edges: bool = False
	add_cross_section_edges: bool = False


def build_graph_from_sections(
	doc_id: str,
	sections: List[Dict[str, Any]],
	chunks: List[Dict[str, Any]],
	config: GraphBuildConfig | None = None,
) -> Any:
	"""Build a hierarchical graph linking document, sections, and chunks.

	Parameters
	----------
	doc_id : str
		Stable identifier for the document root node.
	sections : list[dict]
		Section dictionaries with at least ``section_id`` and ``section_title``.
	chunks : list[dict]
		Chunk dictionaries with at least ``chunk_id``, ``section_id``, ``text``.

	Returns
	-------
	networkx.DiGraph | dict
		A directed graph if ``networkx`` is installed; otherwise a fallback
		dictionary with ``nodes`` and ``edges`` collections.

	Node Types
	----------
	- doc: single root node (``id=doc_id``)
	- section: one per section
	- chunk: one per chunk

	Edge Relations
	--------------
	- doc -> section: ``relation="has_section"``
	- section -> chunk: ``relation="has_chunk"``
	- chunk_i -> chunk_{i+1} within same section: ``relation="next"``

	Node Attributes
	---------------
	- type: "doc" | "section" | "chunk"
	- title: section title (sections only)
	- text: chunk text (chunks only)
	- section_id & doc_id stored on chunk nodes for retrieval convenience.

	Notes
	-----
	The resulting graph enables structured context expansion: given a chunk you
	can traverse to its section siblings or up to the document node efficiently.
	"""
	if not isinstance(doc_id, str) or not doc_id:
		raise TypeError("doc_id must be a non-empty str")
	if not isinstance(sections, list):
		raise TypeError("sections must be a list of dicts")
	if not isinstance(chunks, list):
		raise TypeError("chunks must be a list of dicts")

	if config is None:
		config = GraphBuildConfig()

	if _HAS_NX:
		graph = nx.DiGraph()
		add_node = graph.add_node
		add_edge = graph.add_edge
	else:
		graph = {"nodes": {}, "edges": []}

		def add_node(nid: str, **attrs: Any) -> None:
			graph["nodes"][nid] = attrs

		def add_edge(src: str, dst: str, **attrs: Any) -> None:
			graph["edges"].append((src, dst, attrs))

	# Add document root node
	add_node(doc_id, type="doc", title="", text="")

	# Map section_id -> list of chunk_ids for next-edge linkage
	section_to_chunks: Dict[str, List[str]] = {}

	# Add section nodes + doc->section edges
	for sec in sections:
		if not isinstance(sec, dict):
			continue
		sid = str(sec.get("section_id"))
		if not sid:
			continue
		title = str(sec.get("section_title", ""))
		add_node(sid, type="section", title=title, text="")
		add_edge(doc_id, sid, relation="has_section")
		section_to_chunks.setdefault(sid, [])

	# Add chunk nodes + section->chunk edges
	for ch in chunks:
		if not isinstance(ch, dict):
			continue
		cid = str(ch.get("chunk_id"))
		sid = str(ch.get("section_id"))
		if not cid or not sid:
			continue
		text = str(ch.get("text", ""))
		add_node(
			cid,
			type="chunk",
			text=text,
			title="",  # chunks do not have titles
			section_id=sid,
			doc_id=doc_id,
		)
		if sid in section_to_chunks:
			add_edge(sid, cid, relation="has_chunk")
			section_to_chunks[sid].append(cid)

	# Add sequential next edges within each section (configurable)
	if config.add_next_edges:
		for sid, cid_list in section_to_chunks.items():
			for i in range(len(cid_list) - 1):
				add_edge(cid_list[i], cid_list[i + 1], relation="next")

	# TODO: Add sibling edges within sections when enabled by config.
	# if config.add_sibling_edges:
	#     for sid, cid_list in section_to_chunks.items():
	#         # Placeholder: example structure for future implementation.
	#         # for i, src in enumerate(cid_list):
	#         #     for j, dst in enumerate(cid_list):
	#         #         if i != j:
	#         #             add_edge(src, dst, relation="sibling")
	#         pass

	# TODO: Add cross-section edges when enabled by config.
	# if config.add_cross_section_edges:
	#     # Placeholder: connect chunks across sections based on heuristic/similarity
	#     pass

	return graph


if __name__ == "__main__":  # simple smoke demo
	demo_sections = [
		{"section_id": "s1", "section_title": "Intro"},
		{"section_id": "s2", "section_title": "Body"},
	]
	demo_chunks = [
		{"chunk_id": "s1-0", "section_id": "s1", "text": "Intro text A."},
		{"chunk_id": "s1-1", "section_id": "s1", "text": "Intro text B."},
		{"chunk_id": "s2-0", "section_id": "s2", "text": "Body text A."},
	]
	g = build_graph_from_sections("docX", demo_sections, demo_chunks)
	if _HAS_NX:
		print(f"Nodes: {list(g.nodes(data=True))}")
		print(f"Edges: {list(g.edges(data=True))}")
	else:
		print(g)

