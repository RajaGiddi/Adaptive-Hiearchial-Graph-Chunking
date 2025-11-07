"""Summarization utilities using Google ADK agents.

This module provides thin helpers that build on `ahgc.llm.client` to obtain a
configured ADK agent and generate short, one-sentence summaries for text
segments in the AHGC pipeline.

Two levels are supported:
- summarize_segment: summarize a single string.
- summarize_segments: summarize a list of segment dicts in-place-like, adding a
  "summary" field to each returned dict.

Short-circuit rule
------------------
If a segment's text is very short (< 40 characters), we skip the model call and
use the text itself as the summary to save compute and avoid trivial rewrites.
"""

from __future__ import annotations

from typing import Any, Dict, List
import json

from .client import get_agent, generate_with_agent

__all__ = [
	"summarize_segment",
	"summarize_segments",
]


def summarize_segment(text: str, model: str = "gemini-2.5-flash") -> str:
	"""Summarize a single text segment in one sentence.

	Parameters
	----------
	text : str
		The input segment content to summarize.
	model : str, optional
		ADK model identifier (default is ``"gemini-2.5-flash"``).

	Returns
	-------
	str
		A one-sentence summary of the input text.
	"""
	if not isinstance(text, str):
		raise TypeError("text must be a str")

	# Short-circuit: for very short content, return as-is.
	if len(text.strip()) < 40:
		return text.strip()

	agent = get_agent(
		model=model,
		name="ahgc_summarizer",
		description="Summarizes document segments in one sentence.",
		instruction="You are concise. Provide a single, clear sentence summary.",
	)
	prompt = f"Summarize this segment in one sentence: {text}"
	return generate_with_agent(agent, prompt)


def summarize_segments(
	segments: List[Dict[str, Any]],
	model: str = "gemini-2.5-flash",
	batch_size: int = 5,
) -> List[Dict[str, Any]]:
	"""Summarize a collection of segment dicts.

	Each input dict is expected to contain at least a ``text`` field. The
	returned list mirrors the input order and includes a new ``summary`` field
	in each corresponding dict.

	Parameters
	----------
	segments : list[dict]
		A sequence of segment dictionaries. Each should provide a ``text`` key.
	model : str, optional
		ADK model identifier (default is ``"gemini-2.5-flash"``).

	Returns
	-------
	list[dict]
		New list of segment dictionaries with an added ``summary`` key.
	"""
	if not isinstance(segments, list):
		raise TypeError("segments must be a list of dicts")

	# Determine if any segment actually requires model summarization (>= 40 chars).
	needs_model = any(len(str(seg.get("text", "")).strip()) >= 40 for seg in segments)
	agent = None
	if needs_model:
		# Reuse one agent for efficiency only if needed.
		agent = get_agent(
			model=model,
			name="ahgc_summarizer_batch",
			description="Summarizes multiple segments in one sentence each.",
			instruction="You are concise. Provide single-sentence summaries.",
		)

	# Prepare results container aligned to input order
	results: List[str] = [""] * len(segments)

	def _batch_prompt(items: List[tuple[int, str]]) -> str:
		header = (
			"You will receive multiple segments. Return JSON:\n"
			"{\"summaries\": [{\"segment_id\": \"<id>\", \"summary\": \"<one sentence>\"}]}\n\n"
		)
		lines = [header]
		for idx, text in items:
			lines.append(f"seg_{idx}: {text}")
		return "\n".join(lines)

	def _try_parse_json(text: str) -> Dict[str, Any] | None:
		if not isinstance(text, str) or not text.strip():
			return None
		s = text.strip()
		# Remove code fences if present
		if s.startswith("```"):
			s = s.strip("`\n ")
			# If a language hint remains at start, drop first line
			if "\n" in s:
				first, rest = s.split("\n", 1)
				# If first line doesn't start with '{', drop it
				if not first.strip().startswith("{"):
					s = rest
		# Try direct parse
		try:
			return json.loads(s)
		except Exception:
			# Try to extract the first JSON object substring
			try:
				start = s.find("{")
				end = s.rfind("}")
				if start != -1 and end != -1 and end > start:
					return json.loads(s[start : end + 1])
			except Exception:
				return None
		return None

	# Walk input, short-circuit tiny segments, batch the rest
	pending: List[tuple[int, str]] = []
	batches = 0

	for i, seg in enumerate(segments):
		if not isinstance(seg, dict):
			raise TypeError("each segment must be a dict")
		text = str(seg.get("text", ""))
		trimmed = text.strip()
		if len(trimmed) < 40 or agent is None:
			# Short-circuit or no agent: echo text
			results[i] = trimmed
			continue

		# Defer to batch
		pending.append((i, trimmed))
		if len(pending) >= max(1, int(batch_size)):
			# Process a full batch
			batches += 1
			batch = pending
			pending = []
			prompt = _batch_prompt(batch)
			try:
				resp = generate_with_agent(agent, prompt)
				data = _try_parse_json(resp)
				if not data or not isinstance(data.get("summaries"), list):
					raise ValueError("invalid batch JSON structure")
				mapping = {
					str(item.get("segment_id")): str(item.get("summary", "")).strip()
					for item in data["summaries"]
					if isinstance(item, dict)
				}
				# Fill results, if any missing keys fallback individually
				for idx, _txt in batch:
					key = f"seg_{idx}"
					summary = mapping.get(key, "").strip()
					if summary:
						results[idx] = summary
					else:
						# Fallback to single prompt for this segment
						single_prompt = f"Summarize this segment in one sentence: {_txt}"
						results[idx] = generate_with_agent(agent, single_prompt)
			except Exception:
				# Full batch fallback: per segment
				for idx, _txt in batch:
					single_prompt = f"Summarize this segment in one sentence: {_txt}"
					results[idx] = generate_with_agent(agent, single_prompt)

	# Process any tail batch
	if pending and agent is not None:
		batches += 1
		batch = pending
		prompt = _batch_prompt(batch)
		try:
			resp = generate_with_agent(agent, prompt)
			data = _try_parse_json(resp)
			if not data or not isinstance(data.get("summaries"), list):
				raise ValueError("invalid batch JSON structure")
			mapping = {
				str(item.get("segment_id")): str(item.get("summary", "")).strip()
				for item in data["summaries"]
				if isinstance(item, dict)
			}
			for idx, _txt in batch:
				key = f"seg_{idx}"
				summary = mapping.get(key, "").strip()
				if summary:
					results[idx] = summary
				else:
					single_prompt = f"Summarize this segment in one sentence: {_txt}"
					results[idx] = generate_with_agent(agent, single_prompt)
		except Exception:
			for idx, _txt in batch:
				single_prompt = f"Summarize this segment in one sentence: {_txt}"
				results[idx] = generate_with_agent(agent, single_prompt)

	# Log how many batches we processed (only when agent existed)
	if needs_model:
		print(f"[Summarizer] Processed {batches} batch(es) with batch_size={batch_size}.")

	# Build summarized list of dicts
	summarized: List[Dict[str, Any]] = []
	for i, seg in enumerate(segments):
		summarized.append({**seg, "summary": results[i]})

	return summarized

