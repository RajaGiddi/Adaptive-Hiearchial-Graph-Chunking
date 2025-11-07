"""Minimal LLM protocol abstraction.

This module defines a tiny protocol that AHGC can rely on without depending on
any specific LLM SDK. Concrete backends (e.g., LangChain, OpenAI, etc.) should
provide an adapter that implements this protocol.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal interface for text generation.

    Implementations should make a best-effort to return a plain text string for
    a given prompt. Any backend-specific message shaping should be handled
    internally by the implementation.
    """

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt and return plain text."""
        ...
