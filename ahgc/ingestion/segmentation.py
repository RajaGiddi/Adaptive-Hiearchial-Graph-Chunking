"""Text segmentation utilities for the Adaptive Hierarchical Graph Chunking pipeline.

This initial implementation provides a very simple, deterministic segmentation
strategy suitable as a first step in the ingestion pipeline before more
adaptive / model-informed refinement occurs.

Current strategy:
1. Split the raw input text on double newlines ("\n\n").
2. Strip leading/trailing whitespace from each candidate segment.
3. Skip any empty segments after stripping.
4. Return a list of dictionaries, each containing:
   - segment_id: stable identifier of the form "seg_<index>"
   - text: the raw segment text (post-strip)
   - order: the zero-based positional index

Design notes:
- Deterministic: identical input yields identical segmentation / ordering.
- Minimal: purposefully avoids heuristics so later adaptive passes can build
  hierarchical structure, merge, or further split.
- Extensible: future improvements could accept configuration (e.g., maximum
  length, semantic splitting) while preserving this simple core as a fallback.
"""

from __future__ import annotations

from typing import List, Dict

__all__ = ["segment_document"]


def segment_document(text: str) -> List[Dict[str, object]]:
	"""Segment a raw text string into coarse blocks separated by blank lines.

	Parameters
	----------
	text : str
		The raw input document text.

	Returns
	-------
	list[dict]
		A list of segment dictionaries. Each dict has keys:
		- ``segment_id`` (str): Stable identifier (``seg_<index>``).
		- ``text`` (str): The segment content with surrounding whitespace removed.
		- ``order`` (int): Zero-based index denoting original order.

	Examples
	--------
	>>> segment_document("A\n\nB")
	[{'segment_id': 'seg_0', 'text': 'A', 'order': 0},
	 {'segment_id': 'seg_1', 'text': 'B', 'order': 1}]

	Notes
	-----
	- Empty segments (after stripping) are omitted.
	- The split criterion is a literal double newline ("\n\n"), not a regex.
	"""
	if not isinstance(text, str):  # Defensive: keep deterministic behavior.
		raise TypeError("text must be a str")

	raw_segments = text.split("\n\n") if text else []

	results: List[Dict[str, object]] = []
	for idx, seg in enumerate(raw_segments):
		cleaned = seg.strip()
		if not cleaned:
			continue
		results.append({
			"segment_id": f"seg_{len(results)}",  # sequential after filtering
			"text": cleaned,
			"order": len(results),  # order respects final sequence (same as index here)
		})
	return results


if __name__ == "__main__":  # Simple manual smoke test
	import json
	sample = "A\n\nB\n\n\nC"  # includes an empty segment between B and C
	print(json.dumps(segment_document(sample), indent=2))

