"""Minimal in-memory vector store for local AHGC experiments.

This module provides a tiny wrapper to index chunk texts and perform cosine-
similarity search. It is intended for prototyping only and is not suitable for
production use (no persistence, no ANN, no concurrency control).

Behavior
--------
- Attempts to import `sentence_transformers.SentenceTransformer` for embeddings.
  If unavailable, falls back to a stub embedder that returns a fixed-size zero
  vector and logs a warning exactly once.
- Stores records with keys: ``id``, ``text``, ``metadata``, ``vector`` (np.ndarray)
- `search(query, k)` returns top-k results as: ``[{id, text, metadata, score}]``

Important
---------
- Assume the added records correspond to chunks. Use the pre-existing
  ``chunk_id`` as the ``id`` argument, and include ``section_id``,
  ``section_title``, and ``doc_id`` in ``metadata``. This helps graph-aware
  retrieval later on.
"""

from __future__ import annotations

from typing import Any, Dict, List
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
	from sentence_transformers import SentenceTransformer  # type: ignore
	_HAS_ST = True
except Exception:  # pragma: no cover - only executed when sentence-transformers missing
	SentenceTransformer = None  # type: ignore
	_HAS_ST = False

__all__ = ["VectorStore"]

_STUB_DIM = 384  # default dimension for stub embeddings (MiniLM L6 v2 uses 384)
_STUB_WARNED = False


def _l2_norm(x: np.ndarray) -> float:
	n = float(np.linalg.norm(x))
	return n


class _StubEmbedder:
	"""Fallback embedder that returns a fixed-size zero vector.

	This keeps the rest of the pipeline working even without
	sentence-transformers installed.
	"""

	def __init__(self, dim: int = _STUB_DIM) -> None:
		self.dim = dim

	def encode(self, text: str, convert_to_numpy: bool = True, **_: Any) -> np.ndarray:
		vec = np.zeros(self.dim, dtype=np.float32)
		return vec


class VectorStore:
	"""A tiny, in-memory vector store for chunk-level retrieval.

	Parameters
	----------
	model_name : str
		Name of the sentence-transformers model to load. Ignored if the
		library is unavailable, in which case a zero-vector stub is used.
	"""

	def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
		global _STUB_WARNED
		if _HAS_ST:
			try:
				self._embedder = SentenceTransformer(model_name)
				self._dim = int(getattr(self._embedder, "get_sentence_embedding_dimension", lambda: _STUB_DIM)())
			except Exception as e:  # fallback if model load fails
				if not _STUB_WARNED:
					logger.warning(
						"Failed to load SentenceTransformer(%s): %s. Falling back to zero-vector stub (dim=%d).",
						model_name,
						e,
						_STUB_DIM,
					)
					_STUB_WARNED = True
				self._embedder = _StubEmbedder(_STUB_DIM)
				self._dim = _STUB_DIM
		else:
			if not _STUB_WARNED:
				logger.warning(
					"sentence_transformers not installed; using zero-vector stub embeddings (dim=%d).",
					_STUB_DIM,
				)
				_STUB_WARNED = True
			self._embedder = _StubEmbedder(_STUB_DIM)
			self._dim = _STUB_DIM

		# Internal storage of records
		self._records: List[Dict[str, Any]] = []

	# ---------------------------
	# Internal helpers
	# ---------------------------
	def _embed(self, text: str) -> np.ndarray:
		try:
			vec = self._embedder.encode(text, convert_to_numpy=True)  # type: ignore[attr-defined]
		except TypeError:
			# Some versions might not accept convert_to_numpy arg
			vec = self._embedder.encode(text)  # type: ignore[attr-defined]
			if not isinstance(vec, np.ndarray):
				vec = np.asarray(vec, dtype=np.float32)
		vec = vec.astype(np.float32, copy=False)
		# Normalize defensively to make cosine a dot product for non-zero vectors
		norm = _l2_norm(vec)
		if norm > 0:
			vec = vec / norm
		return vec

	# ---------------------------
	# Public API
	# ---------------------------
	def add_document(self, id: str, text: str, metadata: Dict[str, Any]) -> None:
		"""Add a chunk/document to the store.

		Notes
		-----
		- ``id`` should be the chunk_id.
		- ``metadata`` should include ``section_id``, ``section_title``, and ``doc_id``.
		"""
		if not isinstance(id, str) or not id:
			raise TypeError("id must be a non-empty str")
		if not isinstance(text, str):
			raise TypeError("text must be a str")
		if not isinstance(metadata, dict):
			raise TypeError("metadata must be a dict")

		vec = self._embed(text)
		norm = _l2_norm(vec)
		rec = {
			"id": id,
			"text": text,
			"metadata": metadata,
			"vector": vec,
			"_norm": norm,  # for defensive cosine computation
		}
		self._records.append(rec)

	def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
		"""Return top-k records by cosine similarity.

		Returns
		-------
		list[dict]
			Each dict has keys: ``id``, ``text``, ``metadata``, ``score``.
		"""
		if not self._records:
			return []
		q = self._embed(query)
		q_norm = _l2_norm(q)

		results: List[Dict[str, Any]] = []
		for rec in self._records:
			v = rec["vector"]
			v_norm = float(rec.get("_norm") or _l2_norm(v))
			score: float
			if q_norm == 0.0 or v_norm == 0.0:
				score = 0.0
			else:
				# If both are normalized, this is equivalent to dot product
				score = float(np.dot(q, v) / (q_norm * v_norm))
			results.append({
				"id": rec["id"],
				"text": rec["text"],
				"metadata": rec["metadata"],
				"score": score,
			})

		results.sort(key=lambda r: r["score"], reverse=True)
		return results[: max(0, int(k))]

	# Optional utility
	def __len__(self) -> int:  # pragma: no cover - convenience only
		return len(self._records)

