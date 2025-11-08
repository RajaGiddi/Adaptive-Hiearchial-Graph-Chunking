from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
from eval_retrieval import _load_sentence_transformer, _normalize_for_match, _contains_any, SimpleVectorStore

def main():
    chunker_dir = Path("tmp")
    queries_dir = Path("data/queries")

    sbert = _load_sentence_transformer()
    embed = (lambda t: sbert.encode([t], normalize_embeddings=True)[0].tolist()) if sbert else (lambda t: t.lower())

    # Load all chunks into one store
    all_chunks = []
    for jf in chunker_dir.glob("*_chunkers.jsonl"):
        doc = jf.stem.replace("_chunkers", "")
        for line in jf.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            if isinstance(obj.get("chunks"), list):
                for c in obj["chunks"]:
                    all_chunks.append((doc, obj["method"], c))

    # Evaluate per method globally
    methods = sorted({m for _, m, _ in all_chunks})
    print("method, total_chunks, recall@1")

    for method in methods:
        store = SimpleVectorStore(embed)
        for doc, m, c in all_chunks:
            if m == method:
                store.add(c, {"doc": doc})

        # Combine queries across all docs
        queries = []
        for qf in queries_dir.glob("*.json"):
            data = json.loads(qf.read_text(encoding="utf-8"))
            queries.extend([q for q in data if isinstance(q, dict)])

        hits = 0
        total = 0
        for q in queries:
            qtext, must = q["query"], q.get("must_contain", [])
            total += 1
            res = store.search(qtext, k=1)
            found = any(_contains_any(r["text"], must) for r in res)
            if found:
                hits += 1

        recall = hits / total if total else 0.0
        count = len([1 for _, m, _ in all_chunks if m == method])
        print(f"{method}, {count}, {recall:.4f}")

if __name__ == "__main__":
    main()
