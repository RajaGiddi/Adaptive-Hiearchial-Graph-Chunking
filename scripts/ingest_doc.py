"""End-to-end AHGC ingestion script for a single local document.

Assumptions
-----------
- Google ADK is installed and GOOGLE_API_KEY is set (for summarization/grouping).
- The user has networkx installed (preferred). If not, graph expansion in
  retrieval is limited but basic retrieval still works.

Pipeline
--------
1) Segment the raw document text into segments.
2) Summarize segments with the ADK-backed summarizer.
3) Induce a hierarchy (sections) from summaries with the ADK-backed grouper.
4) Materialize sections by concatenating referenced segments.
5) Rechunk sections into overlapping chunks suitable for embedding.
6) Build a hierarchical graph linking doc → sections → chunks.
7) Index chunks in an in-memory VectorStore.
8) Optionally, accept a query and retrieve a small, graph-expanded context.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List
from dataclasses import dataclass
import json
from pathlib import Path

# Ensure repository root is on sys.path so 'ahgc' imports resolve when running as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:
    nx = None  # type: ignore
    _HAS_NX = False

from ahgc.ingestion.segmentation import segment_document  # noqa: E402
from ahgc.llm.summarizer import summarize_segments  # noqa: E402
from ahgc.llm.grouper import induce_hierarchy  # noqa: E402
from ahgc.chunking.materialize import materialize_sections  # noqa: E402
from ahgc.chunking.rechunk import build_final_chunks  # noqa: E402
from ahgc.graph.builder import build_graph_from_sections  # noqa: E402
from ahgc.indexing.vectorstore import VectorStore  # noqa: E402
from ahgc.retrieval.graph_retrieval import retrieve_with_graph  # noqa: E402
@dataclass
class IngestionReport:
    num_segments: int
    num_sections: int
    num_chunks: int
    num_graph_nodes: int
    num_graph_edges: int



def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _doc_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return stem or "doc"


def _load_env_file_if_present(env_path: Path) -> None:
    """Lightweight .env loader (avoids external dependency).

    Supports lines of the form KEY=VALUE with optional quotes. Skips comments
    and blank lines. Does not override existing environment variables.
    """
    try:
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            key, val = s.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Best-effort; ignore parsing errors
        pass


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest a document with AHGC pipeline")
    parser.add_argument(
        "--path",
        "--input",
        dest="path",
        required=True,
        help="Path to .txt / .md file (alias: --input)",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="ADK LLM model name (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=200, help="Approx tokens per chunk (default: 200)"
    )
    parser.add_argument(
        "--overlap", type=int, default=30, help="Token overlap between chunks (default: 30)"
    )
    args = parser.parse_args(argv)

    path = args.path
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = _read_file(path)
    doc_id = _doc_id_from_path(path)

    # 1) Segment
    segments = segment_document(text)

    # Load .env (if present) before invoking any ADK-backed steps
    _load_env_file_if_present(_REPO_ROOT / ".env")

    # 2) Summarize with ADK (requires GOOGLE_API_KEY)
    summarized = summarize_segments(segments, model=args.model)

    # 3) Induce hierarchy
    hierarchy = induce_hierarchy(summarized_segments=summarized, model=args.model)

    # 4) Materialize sections
    sections = materialize_sections(hierarchy, segments)

    # 5) Rechunk sections
    chunks = build_final_chunks(
        doc_id=doc_id,
        sections=sections,
        max_tokens=int(args.max_tokens),
        overlap=int(args.overlap),
    )

    # 6) Build graph
    graph = build_graph_from_sections(doc_id=doc_id, sections=sections, chunks=chunks)

    # 7) Index chunks in VectorStore
    vs = VectorStore()
    for ch in chunks:
        cid = str(ch["chunk_id"])  # chunk_id is required
        metadata = {
            "section_id": ch.get("section_id"),
            "section_title": ch.get("section_title"),
            "doc_id": doc_id,
        }
        vs.add_document(id=cid, text=str(ch.get("text", "")), metadata=metadata)

    # Summary
    print("=== AHGC Ingestion Summary ===")
    print(f"Segments: {len(segments)}")
    print(f"Sections: {len(sections)}")
    print(f"Chunks:   {len(chunks)}")
    if _HAS_NX and hasattr(graph, "number_of_nodes"):
        print(f"Graph:    nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
        num_nodes = int(graph.number_of_nodes())
        num_edges = int(graph.number_of_edges())
    else:
        try:
            nodes_ct = len(graph.get("nodes", {}))  # type: ignore[attr-defined]
            edges_ct = len(graph.get("edges", []))  # type: ignore[attr-defined]
            print(f"Graph:    nodes={nodes_ct} edges={edges_ct} (dict fallback)")
            num_nodes = int(nodes_ct)
            num_edges = int(edges_ct)
        except Exception:
            print("Graph:    (unknown)")
            num_nodes = 0
            num_edges = 0

    # JSON ingestion report
    report = IngestionReport(
        num_segments=len(segments),
        num_sections=len(sections),
        num_chunks=len(chunks),
        num_graph_nodes=num_nodes,
        num_graph_edges=num_edges,
    )
    print("\n=== Ingestion Report (JSON) ===")
    print(json.dumps(report.__dict__, indent=2))

    # Optional interactive retrieval demo
    try:
        prompt = input("\nEnter a query to retrieve (blank to skip): ").strip()
    except EOFError:
        prompt = ""
    if prompt:
        results = retrieve_with_graph(prompt, vs, graph, k=5)
        print("\n=== Retrieval (top results) ===")
        for i, r in enumerate(results[:5], start=1):
            section_title = r.get("section_title", "")
            chunk_id = r.get("chunk_id", "")
            score = r.get("score", 0.0)
            text_snippet = (r.get("text", "") or "").strip().replace("\n", " ")
            if len(text_snippet) > 160:
                text_snippet = text_snippet[:157] + "..."
            print(f"{i:>2}. [{score:.4f}] {section_title} | {chunk_id}")
            print(f"    {text_snippet}")


if __name__ == "__main__":
    main()
