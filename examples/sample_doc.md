# Sample: Adaptive Hierarchical Graph Chunking

This sample document demonstrates headings, lists, code blocks, and links in Markdown. Replace placeholder text with your project details.

## Overview
Adaptive hierarchical graph chunking splits long documents into semantically coherent chunks, builds a graph over them (nodes = chunks, edges = relationships), and enables multi-scale retrieval.

- Adaptive: chunk sizes vary based on content structure.
- Hierarchical: chunks form levels (sections → paragraphs → sentences).
- Graph: edges capture references, topical proximity, and chronology.

## Quick Start
1. Collect source documents.
2. Parse structure (titles, headings, paragraphs).
3. Create chunks per level with adaptive thresholds.
4. Build inter-chunk edges.
5. Export to your vector store or graph DB.

```python
# minimal, self-contained demo (replace with your pipeline)
from collections import defaultdict

def adaptive_chunks(text, min_len=200, max_len=800):
    buf, chunks = [], []
    for para in text.split("\n\n"):
        buf.append(para.strip())
        size = sum(len(p) for p in buf)
        if size >= min_len and (size >= max_len or para.endswith(".")):
            chunks.append("\n\n".join(buf)); buf = []
    if buf: chunks.append("\n\n".join(buf))
    return chunks

def build_graph(chunks):
    edges = defaultdict(list)
    for i, c in enumerate(chunks):
        if i > 0: edges[i].append(i-1)  # prev
        if i < len(chunks)-1: edges[i].append(i+1)  # next
        # naive topical link by shared keywords
        for j in range(i+1, len(chunks)):
            if len(set(c.lower().split()) & set(chunks[j].lower().split())) > 10:
                edges[i].append(j); edges[j].append(i)
    return edges

sample = """Title: Demo
Intro paragraph about adaptive chunking.

Section A talks about graphs and edges.

Section B references Section A with similar terms like graph, node, and edge."""

chunks = adaptive_chunks(sample)
graph = build_graph(chunks)

print(f"{len(chunks)} chunks")
for i, nbrs in graph.items():
    print(i, "->", sorted(set(nbrs)))
```

## Configuration (example)
```yaml
levels:
  - name: section
    min_len: 600
    max_len: 1500
  - name: paragraph
    min_len: 200
    max_len: 800
linkers:
  - type: adjacency
  - type: keyword_overlap
    threshold: 0.12
exports:
  - type: jsonl
  - type: networkx_pickle
```

## Evaluation
- Coverage: percent of source tokens represented in chunks.
- Redundancy: average overlap between neighboring chunks.
- Connectivity: average node degree in the graph.
- Retrieval@k: relevant chunk found within top-k results.

## Tips
- Start with conservative min_len and increase if recall is low.
- Add headings as anchors; preserve section boundaries.
- Limit cross-links to avoid dense, noisy graphs.

## References
- Concept note: [Graph-based chunking primer](https://example.com/graph-chunking-primer)
- Retrieval best practices: [RAG guide](https://example.com/rag-guide)
- Metrics cookbook: [Evaluation recipes](https://example.com/eval-recipes)