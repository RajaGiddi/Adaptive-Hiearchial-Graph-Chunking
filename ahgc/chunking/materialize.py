"""Materialize section texts from hierarchy and original segments.

This module turns an induced hierarchy JSON and the original segments into
concrete section-level text objects, by concatenating referenced segment texts
in order, separated by a blank line ("\n\n"). It is deterministic and does not
call any LLMs.

Primary function
----------------
materialize_sections(hierarchy, segments) -> list[dict]
	For each section in ``hierarchy["sections"]``, collect the text of the
	referenced ``segment_ids`` in the original order and join them with two
	newlines. Return a list of dicts with keys: ``section_id``, ``section_title``,
	``text``.

Notes
-----
- If a segment id from the hierarchy is missing in the provided ``segments``,
  it is skipped gracefully; order is preserved among those found.
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__ = ["materialize_sections"]


def materialize_sections(
	hierarchy: Dict[str, Any],
	segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
	"""Materialize section texts by concatenating referenced segments.

	Parameters
	----------
	hierarchy : dict
		Dictionary with key ``sections``: list of dicts with
		``section_id``, ``section_title``, and ``segment_ids`` (list of str).
	segments : list[dict]
		Sequence of segment dictionaries with at least ``segment_id`` and
		``text`` fields.

	Returns
	-------
	list[dict]
		A list of section dicts with keys: ``section_id``, ``section_title``,
		``text`` where ``text`` is the concatenation ("\n\n"-joined) of the
		referenced segment texts in original order. Missing segments are skipped.
	"""
	if not isinstance(hierarchy, dict) or "sections" not in hierarchy:
		raise TypeError("hierarchy must be a dict with a 'sections' list")
	if not isinstance(segments, list):
		raise TypeError("segments must be a list of dicts")

	# Map segment_id -> text for quick lookup
	by_id: Dict[str, str] = {}
	for seg in segments:
		if not isinstance(seg, dict):
			raise TypeError("each segment must be a dict")
		sid = str(seg.get("segment_id"))
		text = str(seg.get("text", ""))
		if sid:
			by_id[sid] = text

	sections = hierarchy.get("sections") or []
	if not isinstance(sections, list):
		raise TypeError("hierarchy['sections'] must be a list")

	out: List[Dict[str, Any]] = []
	for idx, sec in enumerate(sections):
		if not isinstance(sec, dict):
			continue
		sec_id = str(sec.get("section_id", f"s{idx+1}"))
		sec_title = str(sec.get("section_title", "Untitled"))
		seg_ids = list(map(str, sec.get("segment_ids", [])))

		texts: List[str] = []
		for sid in seg_ids:
			text = by_id.get(sid)
			if not text:
				continue
			clean = text.strip()
			if clean:
				texts.append(clean)

		out.append({
			"section_id": sec_id,
			"section_title": sec_title,
			"text": "\n\n".join(texts),
		})

	return out


if __name__ == "__main__":  # basic smoke test
	segments = [
		{"segment_id": "seg_0", "text": "A"},
		{"segment_id": "seg_1", "text": "B"},
	]
	hierarchy = {
		"sections": [
			{
				"section_id": "s1",
				"section_title": "All",
				"segment_ids": ["seg_0", "seg_1"],
			}
		]
	}
	import json

	print(json.dumps(materialize_sections(hierarchy, segments), indent=2))

