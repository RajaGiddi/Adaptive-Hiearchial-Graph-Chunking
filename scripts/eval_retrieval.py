"""Evaluate retrieval quality for chunking methods on a single document.

Given:
- --input: the original text/markdown file
- --jsonl: JSONL output produced by scripts/evaluate_chunkers.py (one method per line)
- --queries: optional JSON or YAML file containing a list of {"query": str, "must_contain": [str, ...]}

For each method in the JSONL with a non-empty "chunks" array:
- Build a tiny in-memory vector store (per method)
- Insert each chunk as a separate doc with metadata {"chunk_id": i}
- For each query, run top-k search and check if ANY retrieved chunk contains ANY of the
  must_contain strings (case-insensitive). Record 1 for hit, 0 for miss.
- Compute recall@k = hits / total_queries and report it.

Outputs a compact table to stdout and optionally writes JSONL with the per-method results.

Notes / TODOs:
- If sentence_transformers is unavailable or model fails to load, we fall back to a deterministic
  hashing-based embedding. This keeps behavior stable and repeatable.
- Later we can plug in real datasets like BEIR/LoTTE here.
"""

from __future__ import annotations

import argparse
import re
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ------------------------------- Embeddings -------------------------------

class _StubEmbedder:
    """Deterministic hashing-based embedding as a fallback.

    Produces a fixed-length dense vector by hashing tokens into buckets.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = int(dim)

    def __call__(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for tok in _simple_tokenize(text):
            idx = hash(tok) % self.dim
            v[idx] += 1.0
        return v


def _simple_tokenize(text: str) -> List[str]:
    out: List[str] = []
    buf: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out


def _load_sentence_transformer() -> Any | None:
    """Try to load a SentenceTransformer model; return None if not available.

    We attempt a widely available small model. If any error occurs (package missing,
    model download blocked, etc.), we return None so the caller can fall back.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            # Warm-up simple encode to catch device/ops issues early
            _ = model.encode(["warmup"], normalize_embeddings=True)
            return model
        except Exception as e:
            print(f"[warn] failed to initialize SentenceTransformer: {e}", file=sys.stderr)
            return None
    except Exception:
        return None


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


# ------------------------------ Vector Store ------------------------------

class SimpleVectorStore:
    def __init__(self, embed_fn):
        self._embed = embed_fn
        self.docs: List[Tuple[str, Dict[str, Any]]] = []
        self.vecs: List[List[float]] = []

    def add(self, text: str, metadata: Dict[str, Any]) -> None:
        v = self._embed(text)
        self.docs.append((text, metadata))
        self.vecs.append(v)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        qv = self._embed(query)
        scored: List[Tuple[float, int]] = []  # (score, idx)
        for i, v in enumerate(self.vecs):
            s = _cosine_sim(qv, v)
            scored.append((s, i))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for s, i in scored[: max(1, int(k))]:
            text, meta = self.docs[i]
            out.append({"text": text, "metadata": meta, "score": float(s)})
        return out


# --------------------------------- IO ------------------------------------

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception as e:
                print(f"[warn] skipping invalid JSONL line: {e}", file=sys.stderr)
                continue
    return out


def _load_queries(path: Path | None, source_text: str) -> List[Dict[str, Any]]:
    if path is None:
        # Hardcoded demo queries derived from common concepts in the example doc
        return [
            {"query": "what is adaptive hierarchical graph chunking", "must_contain": ["adaptive", "chunk"]},
            {"query": "how do we build the graph", "must_contain": ["graph", "edge"]},
            {"query": "show me the configuration example", "must_contain": ["levels", "linkers"]},
            {"query": "what are evaluation metrics", "must_contain": ["coverage", "retrieval"]},
        ]

    # Try JSON first
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and "query" in d and "must_contain" in d]
    except Exception:
        pass

    # Try YAML if available
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and "query" in d and "must_contain" in d]
    except Exception:
        print("[warn] failed to parse queries file; using defaults", file=sys.stderr)

    return [
        {"query": "what is adaptive hierarchical graph chunking", "must_contain": ["adaptive", "chunk"]},
        {"query": "how do we build the graph", "must_contain": ["graph", "edge"]},
        {"query": "show me the configuration example", "must_contain": ["levels", "linkers"]},
        {"query": "what are evaluation metrics", "must_contain": ["coverage", "retrieval"]},
    ]


def _normalize_for_match(text: str) -> str:
    """Normalize text for substring matching:
    - lowercase
    - normalize hyphens/underscores to spaces
    - strip punctuation (keep letters, numbers, and spaces)
    - collapse whitespace
    """
    s = str(text).lower()
    s = re.sub(r"[-_]+", " ", s)
    # keep only a-z0-9 and whitespace; map others to spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _contains_any(haystack: str, needles: List[str]) -> bool:
    s = _normalize_for_match(haystack)
    for n in needles:
        if not n:
            continue
        if _normalize_for_match(n) in s:
            return True
    return False


# ------------------------------ Evaluation -------------------------------

def _evaluate_method(method: str, chunks: List[str], queries: List[Dict[str, Any]], k: int,
                     embedder: Any) -> Dict[str, Any]:
    store = SimpleVectorStore(embedder)
    for i, c in enumerate(chunks):
        if not isinstance(c, str):
            c = str(c)
        store.add(c, {"chunk_id": i})

    hits = 0
    total = 0
    for q in queries:
        qtext = str(q.get("query", "")).strip()
        must = q.get("must_contain", [])
        if not qtext or not isinstance(must, list):
            continue
        total += 1
        results = store.search(qtext, k=k)
        found = False
        for r in results:
            if _contains_any(r.get("text", ""), must):
                found = True
                break
        if found:
            hits += 1

    recall = (hits / total) if total > 0 else 0.0
    return {
        "method": method,
        "recall_at_k": recall,
        "num_chunks": len(chunks),
        "k": k,
        "total_queries": total,
        "hits": hits,
    }


def _print_table(rows: List[Dict[str, Any]], k: int) -> None:
    print(f"method, recall@{k}, num_chunks")
    for r in rows:
        print(f"{r['method']}, {round(float(r.get('recall_at_k', 0.0)), 4)}, {r.get('num_chunks', 0)}")


# --------------------------------- Main ----------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality per chunking method")
    parser.add_argument("--input", required=True, help="Path to the original text/markdown file")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL output of evaluate_chunkers.py")
    parser.add_argument("--queries", help="Optional JSON or YAML file with queries and must_contain")
    parser.add_argument("--k", type=int, default=5, help="Top-K for retrieval (default: 5)")
    parser.add_argument("--output", help="Optional JSONL output path for results")
    args = parser.parse_args(argv)

    text_path = Path(args.input)
    if not text_path.exists():
        print(f"File not found: {text_path}", file=sys.stderr)
        sys.exit(1)
    source_text = text_path.read_text(encoding="utf-8")

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"File not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)
    lines = _read_jsonl(jsonl_path)

    # Build methods list with chunks
    methods: List[Tuple[str, List[str]]] = []
    for obj in lines:
        method = str(obj.get("method", "")).strip()
        chunks = obj.get("chunks")
        if not method or not isinstance(chunks, list) or not chunks:
            continue
        # normalize chunk texts
        norm_chunks = [str(c) for c in chunks if str(c).strip()]
        if not norm_chunks:
            continue
        methods.append((method, norm_chunks))

    # Load or synthesize queries
    qpath = Path(args.queries) if args.queries else None
    queries = _load_queries(qpath, source_text)

    # Initialize embedder
    sbert = _load_sentence_transformer()
    if sbert is not None:
        print("[info] using SentenceTransformer embeddings", file=sys.stderr)

        def embed_fn(text: str) -> List[float]:  # type: ignore
            vec = sbert.encode([text], normalize_embeddings=True)[0]
            # Convert to plain list to avoid downstream numpy dependency
            if hasattr(vec, "tolist"):
                return vec.tolist()  # type: ignore
            return list(vec)

    else:
        print("[warn] sentence_transformers unavailable; using deterministic hashing embedder", file=sys.stderr)
        stub = _StubEmbedder(dim=512)

        def embed_fn(text: str) -> List[float]:  # type: ignore
            return stub(text)

    # Evaluate per method
    results: List[Dict[str, Any]] = []
    for method, chunks in methods:
        res = _evaluate_method(method, chunks, queries, k=int(args.k), embedder=embed_fn)
        results.append(res)

    # Print compact table
    _print_table(results, k=int(args.k))

    # Optional JSONL output
    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
