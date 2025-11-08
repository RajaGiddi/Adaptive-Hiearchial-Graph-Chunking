from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict

RETRIEVAL_DIR = Path("tmp")

def main():
    rows = []
    for path in RETRIEVAL_DIR.glob("*_retrieval.jsonl"):
        doc_name = path.name.replace("_retrieval.jsonl", "")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                obj["doc"] = doc_name
                rows.append(obj)

    # method -> list of recalls
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(float(r.get("recall_at_k") or r.get("recall") or 0.0))

    print("method, docs, avg_recall, min_recall, max_recall")
    for method, vals in sorted(by_method.items()):
        avg_rec = sum(vals) / len(vals)
        print(f"{method}, {len(vals)}, {avg_rec:.4f}, {min(vals):.4f}, {max(vals):.4f}")

if __name__ == "__main__":
    main()
