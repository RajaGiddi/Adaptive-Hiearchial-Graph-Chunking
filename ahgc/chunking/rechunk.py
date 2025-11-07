"""Re-chunk section texts into retrieval-friendly pieces.

This module provides helper functions to split section-level texts produced by
materialization into smaller overlapping chunks suitable for retrieval and
embedding. It offers a pluggable tokenization hook, automatic short-circuit for
small texts, and both eager (list-returning) and streaming (iterator) APIs.

Functions
---------
rechunk_section(section, max_tokens=500, overlap=50, tokenize=default_tokenize)
	Split a single section's text into overlapping token windows. Returns chunk
	dicts with ``chunk_id``, ``section_id``, ``section_title``, ``text``.

iter_rechunk_section(section, max_tokens=500, overlap=50, tokenize=default_tokenize)
	Generator version of ``rechunk_section`` that yields chunks lazily.

build_final_chunks(doc_id, sections, max_tokens=500, overlap=50, tokenize=default_tokenize)
	Re-chunk all sections eagerly and flatten into a single list. Chunk IDs are
	prefixed with the document id (``{doc_id}-{section_id}-{index}``).

iter_final_chunks(doc_id, sections, max_tokens=500, overlap=50, tokenize=default_tokenize)
	Streaming version of ``build_final_chunks`` yielding chunks one-by-one.

Optional future extension
-------------------------
You can pass a real tokenizer (e.g., from ``tiktoken`` or ``google-generativeai``)
via the ``tokenize`` parameter without changing chunking semantics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Callable, Iterator, Sequence
import re

TokenizeFn = Callable[[str], List[str]]

__all__ = [
	"TokenizeFn",
	"default_tokenize",
	"estimate_tokens_by_chars",
	"rechunk_section",
	"iter_rechunk_section",
	"build_final_chunks",
	"iter_final_chunks",
]


def default_tokenize(text: str) -> List[str]:
	"""Default whitespace tokenization.

	Parameters
	----------
	text : str
		Raw input text.

	Returns
	-------
	list[str]
		List of whitespace-delimited tokens (contiguous non-whitespace runs).
	"""
	if not text:
		return []
	return re.findall(r"\S+", text)


def _approx_detokenize(tokens: Sequence[str]) -> str:
	"""Convert tokens back to a plain string, separated by a single space."""
	return " ".join(tokens)


def estimate_tokens_by_chars(text: str, chars_per_token: int = 4) -> int:
	"""Estimate token count from raw text length.

	Parameters
	----------
	text : str
		Raw text to estimate.
	chars_per_token : int, optional
		Heuristic divisor; defaults to 4 (common rough estimate for English).

	Returns
	-------
	int
		Estimated number of tokens.
	"""
	if chars_per_token <= 0:
		chars_per_token = 4
	return max(0, (len(text) + chars_per_token - 1) // chars_per_token)


def rechunk_section(
	section: Dict[str, Any],
	max_tokens: int = 500,
	overlap: int = 50,
	tokenize: TokenizeFn = default_tokenize,
) -> List[Dict[str, str]]:
	"""Split a section's text into overlapping chunks (eager version).

	Parameters
	----------
	section : dict
		Section object with at least ``section_id``, ``section_title``, ``text``.
	max_tokens : int, optional
		Maximum tokens per chunk (default 500).
	overlap : int, optional
		Number of overlapping tokens between consecutive chunks (default 50).
	tokenize : TokenizeFn, optional
		Tokenization function. Defaults to :func:`default_tokenize`. Provide a
		custom tokenizer (e.g., tiktoken) for model-aligned chunking.

	Returns
	-------
	list[dict[str, str]]
		Chunk dictionaries with keys: ``chunk_id``, ``section_id``,
		``section_title``, ``text``.

	Notes
	-----
	- If ``overlap`` is invalid (negative or >= ``max_tokens``), it will be
	  clamped into ``[0, max_tokens - 1]``.
	- If using the default tokenizer and the estimated token count is <=
	  ``max_tokens``, returns a single chunk immediately.
	- Empty / whitespace-only text returns an empty list.
	"""
	if not isinstance(section, dict):
		raise TypeError("section must be a dict")
	sec_id = str(section.get("section_id", "s"))
	sec_title = str(section.get("section_title", ""))
	text = str(section.get("text", ""))
	if not text.strip():
		return []

	if max_tokens <= 0:
		raise ValueError("max_tokens must be > 0")
	if overlap < 0:
		overlap = 0
	if overlap >= max_tokens:
		overlap = max_tokens - 1

	# Short-circuit for small texts when using default tokenizer.
	if tokenize is default_tokenize and estimate_tokens_by_chars(text) <= max_tokens:
		return [
			{
				"chunk_id": f"{sec_id}-0",
				"section_id": sec_id,
				"section_title": sec_title,
				"text": text.strip(),
			}
		]

	tokens = tokenize(text)
	if not tokens:
		return []

	chunks: List[Dict[str, str]] = []
	start = 0
	index = 0
	step = max_tokens - overlap

	while start < len(tokens):
		end = min(start + max_tokens, len(tokens))
		window = tokens[start:end]
		chunk_text = _approx_detokenize(window)
		chunks.append(
			{
				"chunk_id": f"{sec_id}-{index}",
				"section_id": sec_id,
				"section_title": sec_title,
				"text": chunk_text,
			}
		)
		index += 1
		if end == len(tokens):
			break
		start += step

	return chunks


def iter_rechunk_section(
	section: Dict[str, Any],
	max_tokens: int = 500,
	overlap: int = 50,
	tokenize: TokenizeFn = default_tokenize,
) -> Iterator[Dict[str, str]]:
	"""Yield chunks for a section lazily.

	Parameters
	----------
	section : dict
		Section object with ``section_id``, ``section_title``, ``text``.
	max_tokens : int, optional
		Maximum tokens per chunk.
	overlap : int, optional
		Token overlap between consecutive chunks.
	tokenize : TokenizeFn, optional
		Tokenization function.

	Yields
	------
	dict[str, str]
		Chunk dictionaries with keys: ``chunk_id``, ``section_id``,
		``section_title``, ``text``.
	"""
	if not isinstance(section, dict):
		raise TypeError("section must be a dict")
	sec_id = str(section.get("section_id", "s"))
	sec_title = str(section.get("section_title", ""))
	text = str(section.get("text", ""))
	if not text.strip():
		return  # nothing to yield

	if max_tokens <= 0:
		raise ValueError("max_tokens must be > 0")
	if overlap < 0:
		overlap = 0
	if overlap >= max_tokens:
		overlap = max_tokens - 1

	# Short-circuit small texts
	if tokenize is default_tokenize and estimate_tokens_by_chars(text) <= max_tokens:
		yield {
			"chunk_id": f"{sec_id}-0",
			"section_id": sec_id,
			"section_title": sec_title,
			"text": text.strip(),
		}
		return

	tokens = tokenize(text)
	if not tokens:
		return

	start = 0
	index = 0
	step = max_tokens - overlap
	while start < len(tokens):
		end = min(start + max_tokens, len(tokens))
		window = tokens[start:end]
		chunk_text = _approx_detokenize(window)
		yield {
			"chunk_id": f"{sec_id}-{index}",
			"section_id": sec_id,
			"section_title": sec_title,
			"text": chunk_text,
		}
		index += 1
		if end == len(tokens):
			break
		start += step


def build_final_chunks(
	doc_id: str,
	sections: List[Dict[str, Any]],
	max_tokens: int = 500,
	overlap: int = 50,
	tokenize: TokenizeFn = default_tokenize,
) -> List[Dict[str, str]]:
	"""Create final flat chunk list eagerly by re-chunking all sections.

	Parameters
	----------
	doc_id : str
		Stable identifier for the source document; used to prefix chunk ids.
	sections : list[dict]
		Section objects (``section_id``, ``section_title``, ``text``).
	max_tokens : int, optional
		Maximum tokens per chunk.
	overlap : int, optional
		Overlapping tokens between consecutive chunks.
	tokenize : TokenizeFn, optional
		Tokenization function.

	Returns
	-------
	list[dict[str, str]]
		Flat list of chunk dictionaries with keys: ``chunk_id``, ``section_id``,
		``section_title``, ``text``. Chunk IDs are prefixed with ``doc_id``.
	"""
	if not isinstance(doc_id, str) or not doc_id:
		raise TypeError("doc_id must be a non-empty str")
	if not isinstance(sections, list):
		raise TypeError("sections must be a list of dicts")

	final: List[Dict[str, str]] = []
	for sec in sections:
		parts = rechunk_section(sec, max_tokens=max_tokens, overlap=overlap, tokenize=tokenize)
		# Prefix doc_id into chunk_id
		for p in parts:
			p["chunk_id"] = f"{doc_id}-{p['chunk_id']}"
			final.append(p)
	return final


def iter_final_chunks(
	doc_id: str,
	sections: List[Dict[str, Any]],
	max_tokens: int = 500,
	overlap: int = 50,
	tokenize: TokenizeFn = default_tokenize,
) -> Iterator[Dict[str, str]]:
	"""Yield all chunks across sections lazily.

	Parameters
	----------
	doc_id : str
		Document identifier used to prefix chunk ids.
	sections : list[dict]
		Section objects.
	max_tokens : int, optional
		Maximum tokens per chunk.
	overlap : int, optional
		Overlapping tokens between consecutive chunks.
	tokenize : TokenizeFn, optional
		Tokenization function.

	Yields
	------
	dict[str, str]
		Chunk dictionaries as produced by :func:`iter_rechunk_section` with
		chunk_id prefixed by ``doc_id``.
	"""
	if not isinstance(doc_id, str) or not doc_id:
		raise TypeError("doc_id must be a non-empty str")
	if not isinstance(sections, list):
		raise TypeError("sections must be a list of dicts")
	for sec in sections:
		for chunk in iter_rechunk_section(sec, max_tokens=max_tokens, overlap=overlap, tokenize=tokenize):
			chunk["chunk_id"] = f"{doc_id}-{chunk['chunk_id']}"
			yield chunk

