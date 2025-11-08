#!/usr/bin/env python3
"""
Enhanced visualization: Recall@1 vs #Chunks with Pareto frontier and styling.

Usage:
    python scripts/plot_recall_vs_chunks.py \
        --csv tmp/global_retrieval.csv \
        --out tmp/recall_vs_chunks_pareto.png
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_data(csv_path: Path):
    methods, chunks, recalls = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            methods.append(row["method"])
            chunks.append(int(float(row["total_chunks"])))
            recalls.append(float(row["recall@1"]))
    return methods, np.array(chunks), np.array(recalls)


def compute_pareto_frontier(chunks, recalls):
    """Return indices on the Pareto frontier (low chunks, high recall)."""
    points = sorted(zip(chunks, recalls), key=lambda x: x[0])
    frontier = []
    best_recall = 0.0
    for x, y in points:
        if y >= best_recall:
            frontier.append((x, y))
            best_recall = y
    return np.array(frontier)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", help="Output image path (.png)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    methods, chunks, recalls = load_data(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter points
    ax.scatter(chunks, recalls, s=70, color="C0", alpha=0.7, label="Methods")

    # Highlight AHGC
    for m, x, y in zip(methods, chunks, recalls):
        if m.lower() == "ahgc":
            ax.scatter([x], [y], s=120, color="red", label="AHGC", zorder=5)
            ax.annotate("AHGC", (x, y), textcoords="offset points", xytext=(6, 6),
                        fontsize=10, color="red", fontweight="bold")
        else:
            ax.annotate(m, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, alpha=0.85)

    # Pareto frontier
    frontier = compute_pareto_frontier(chunks, recalls)
    ax.plot(frontier[:, 0], frontier[:, 1], linestyle="--", color="black",
            linewidth=1.2, alpha=0.7, label="Pareto frontier")

    # Axis styling
    ax.set_xlabel("# Chunks (lower is better)", fontsize=11)
    ax.set_ylabel("Recall@1 (higher is better)", fontsize=11)
    ax.set_title("Chunking Methods: Recall@1 vs. #Chunks", fontsize=13, pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    plt.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=300)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
