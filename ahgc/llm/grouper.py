"""Hierarchy induction utilities using Google ADK agents.

This module infers higher-level section groupings from a list of summarized
segments produced earlier in the AHGC pipeline. It leverages an ADK LLM agent
to propose a JSON structure describing sections and the segment IDs they
contain.

Core function
-------------
induce_hierarchy(summarized_segments, model="gemini-2.5-flash") -> dict
	Builds a prompt enumerating each segment as:
		``<segment_id>: <summary>``
	Requests the model to group segments into ordered sections and return ONLY
	JSON of the form:
	``{"sections": [{"section_id": "s1", "section_title": "Title", "segment_ids": ["seg_0", ...]}]}``

Robustness
----------
If the model output cannot be parsed as valid JSON or fails validation (missing
segments, duplicates, order issues), the function falls back to a deterministic
single-section hierarchy that includes all segments in original order.

Validation Rules
----------------
- All input segment IDs must appear exactly once overall.
- Order of segment IDs inside sections must respect original ordering.
- No unknown segment IDs are allowed.
- Each section entry must have: section_id, section_title, segment_ids (list).

The LLM prompt is designed to elicit a clean JSON-only response, but defensive
parsing handles extra tokens gracefully.
"""

from __future__ import annotations

from typing import Any, Dict, List
import json
import re

from .client import get_agent, generate_with_agent

__all__ = ["induce_hierarchy"]


def _strip_code_fences(text: str) -> str:
	text = str(text).strip()
	if text.startswith("```"):
		parts = text.split("```")
		# return the middle content if present
		if len(parts) >= 3:
			return parts[1].strip()
	return text


def induce_hierarchy(
	summarized_segments: List[Dict[str, Any]],
	model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
	"""Infer a hierarchical section structure from summarized segments.

	Parameters
	----------
	summarized_segments : list[dict]
		Each dict should contain at least ``segment_id`` and ``summary`` keys.
	model : str, optional
		ADK model identifier to use (default ``"gemini-2.5-flash"``).

	Returns
	-------
	dict
		Hierarchy dictionary of the form:
		``{"sections": [{"section_id": str, "section_title": str, "segment_ids": [str, ...]}, ...]}``
		Guaranteed to reference all input segment IDs exactly once.
	"""
	if not isinstance(summarized_segments, list):
		raise TypeError("summarized_segments must be a list of dicts")

	# Collect segment ids and summaries; default summary falls back to text.
	lines: List[str] = []
	segment_ids: List[str] = []
	for seg in summarized_segments:
		if not isinstance(seg, dict):
			raise TypeError("Each summarized segment must be a dict")
		sid = str(seg.get("segment_id"))
		if not sid:
			raise ValueError("Segment dict missing 'segment_id'")
		summary = str(seg.get("summary", seg.get("text", ""))).strip()
		lines.append(f"{sid}: {summary}")
		segment_ids.append(sid)

	# Build prompt.
	listing = "\n".join(lines)
	prompt = (
		"You are given a list of segments with ids and one-sentence summaries.\n"
		f"{listing}\n\n"
		"Group these segments into higher-level sections, preserving original order.\n"
		"Return only JSON of the form:\n"
		'{"sections":[{"section_id":"s1","section_title":"Title","segment_ids":["seg_0","seg_1"]}]}'
		"\nConstraints:\n"
		"- Each segment id appears exactly once in exactly one section.\n"
		"- Preserve original order of segments globally (no reordering).\n"
		"- Use concise informative section_title values.\n"
		"- Return ONLY JSON (no commentary).\n"
	)

	# Acquire agent and get response text.
	agent = get_agent(
		model=model,
		name="ahgc_grouper",
		description="Induces hierarchical sections from summarized segments.",
		instruction=(
			"You group related segments preserving order and output strict JSON conforming to given schema."
		),
	)
	raw_response = generate_with_agent(agent, prompt)

	# Strip Markdown code fences before parsing JSON
	cleaned = _strip_code_fences(raw_response)

	hierarchy = _parse_and_validate_hierarchy(cleaned, segment_ids)
	if hierarchy is None:
		return _fallback_hierarchy(segment_ids)
	return hierarchy


def _parse_and_validate_hierarchy(response_text: str, original_ids: List[str]) -> Dict[str, Any] | None:
	"""Extract JSON from model response and validate against constraints.

	Returns a valid hierarchy dict or None on failure.
	"""
	if not isinstance(response_text, str) or not response_text.strip():
		return None

	# Attempt direct parse first.
	json_text = response_text.strip()
	parsed = _try_json_loads(json_text)
	if parsed is None:
		# Try to extract first JSON object via regex (greedy to last closing brace).
		match = re.search(r"\{.*\}$", response_text.strip(), re.DOTALL)
		if match:
			parsed = _try_json_loads(match.group(0))
	if parsed is None:
		# Attempt to find the first '{' and last '}' inclusive.
		first = response_text.find("{")
		last = response_text.rfind("}")
		if first != -1 and last != -1 and last > first:
			parsed = _try_json_loads(response_text[first : last + 1])
	if parsed is None:
		return None

	# Basic shape check.
	if not isinstance(parsed, dict) or "sections" not in parsed:
		return None
	sections = parsed.get("sections")
	if not isinstance(sections, list) or not sections:
		return None

	# Validate each section.
	seen_ids: List[str] = []
	for sec in sections:
		if not isinstance(sec, dict):
			return None
		if not {"section_id", "section_title", "segment_ids"}.issubset(sec.keys()):
			return None
		seg_ids = sec.get("segment_ids")
		if not isinstance(seg_ids, list) or not all(isinstance(x, str) for x in seg_ids):
			return None
		seen_ids.extend(seg_ids)

	# Must match original IDs exactly and preserve global order.
	if set(seen_ids) != set(original_ids):
		return None
	# Preserve order: the sequence of seen_ids must equal original_ids.
	if seen_ids != original_ids:
		return None

	# Ensure uniqueness of section_ids.
	section_ids = [sec.get("section_id") for sec in sections]
	if len(section_ids) != len(set(section_ids)):
		return None

	return {"sections": sections}


def _try_json_loads(text: str) -> Dict[str, Any] | List[Any] | None:
	try:
		return json.loads(text)
	except Exception:
		return None


def _fallback_hierarchy(segment_ids: List[str]) -> Dict[str, Any]:
	return {
		"sections": [
			{
				"section_id": "s1",
				"section_title": "All Segments",
				"segment_ids": segment_ids[:],
			}
		]
	}


if __name__ == "__main__":  # Simple manual demonstration using mock segments.
	mock_segments = [
		{"segment_id": "seg_0", "summary": "Introduction to topic"},
		{"segment_id": "seg_1", "summary": "Detailed explanation"},
		{"segment_id": "seg_2", "summary": "Edge cases and nuances"},
	]
	try:
		print(json.dumps(induce_hierarchy(mock_segments), indent=2))
	except Exception as e:  # noqa: BLE001
		# If API key missing or ADK not installed, show fallback explicitly.
		print("[grouper demo] Falling back due to error:", e)
		print(json.dumps(_fallback_hierarchy([m["segment_id"] for m in mock_segments]), indent=2))

