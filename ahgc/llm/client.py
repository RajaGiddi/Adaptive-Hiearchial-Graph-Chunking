"""LLM client utilities wrapping LangChain's Google Generative AI backend.

This module replaces the previous Google ADK integration with a lightweight
LangChain-based client. It exposes the same public interface so the rest of the
AHGC pipeline (summarizer, grouper) works without changes.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from .interface import LLMClient

__all__ = [
    "get_agent",
    "generate_with_agent",
    "LangChainLLMClient",
    "get_llm_client",
]


def get_agent(
    model: str,
    name: str,
    description: str,
    instruction: str = "You are a helpful LLM used for document structuring.",
) -> Any:
    """Create and return a LangChain ChatGoogleGenerativeAI instance.

    Reads GOOGLE_API_KEY from the environment. Attaches the provided
    instruction as a private attribute for use in generate_with_agent.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY environment variable. Set it before using the LLM."
        )

    llm = ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
    )
    # Stash instruction for later use in generate_with_agent
    setattr(llm, "_ahgc_instruction", instruction or "")
    print(f"[LangChain] Loaded ChatGoogleGenerativeAI model: {model}", file=sys.stderr)
    return llm


def _extract_text(response: Any) -> Optional[str]:
    """Extract a plain string from LangChain AIMessage-like responses.

    The ChatGoogleGenerativeAI .invoke returns an AIMessage with a .content
    attribute (string or list). We normalize to a plain string.
    """
    print("[LangChain RESPONSE - raw]", repr(response), file=sys.stderr)

    if response is None:
        return None

    # Direct string (unlikely, but safe)
    if isinstance(response, str):
        s = response.strip()
        return s or None

    # LangChain AIMessage: prefer .content, fallback to .text
    for attr in ("content", "text"):
        if hasattr(response, attr):
            val = getattr(response, attr)
            if isinstance(val, str):
                s = val.strip()
                return s or None
            # Sometimes .content can be a list of parts; join any textual bits
            if isinstance(val, (list, tuple)):
                parts = []
                for p in val:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict) and isinstance(p.get("text"), str):
                        parts.append(p["text"]) 
                s = " ".join(parts).strip()
                return s or None

    # Last resort: string representation
    try:
        s = str(response)
        return s if s and s != object.__repr__(response) else None
    except Exception:
        return None


def generate_with_agent(agent: Any, user_message: str) -> str:
    """Invoke the LLM and return a plain text result.

    Uses agent.invoke(...) and extracts response.content/text.
    """
    if not isinstance(user_message, str):
        raise TypeError("user_message must be a str")

    # Build messages with system instruction if available
    instruction = getattr(agent, "_ahgc_instruction", "")
    payload: Any
    if instruction:
        payload = [SystemMessage(content=instruction), HumanMessage(content=user_message)]
    else:
        payload = user_message

    try:
        resp = agent.invoke(payload)
        text = _extract_text(resp)
        if text:
            print("[LangChain RESPONSE]", text[:500], file=sys.stderr)
            return text
        raise RuntimeError("No text content in LangChain response.")
    except Exception as e:
        print(f"[LangChain ERROR] invoke failed: {e!r}", file=sys.stderr)
        raise RuntimeError(
            f"LangChain invocation failed: {e!r}. Check network and GOOGLE_API_KEY."
        ) from e


class LangChainLLMClient:
    """Adapter implementing the LLMClient protocol for a LangChain chat model.

    It wraps a LangChain-compatible chat model (agent) and exposes a simple
    generate(prompt) -> str API that returns plain text by delegating to
    generate_with_agent.
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def generate(self, prompt: str) -> str:  # type: ignore[override]
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        return generate_with_agent(self._agent, prompt)


def get_llm_client(
    model: str,
    name: str,
    description: str,
    instruction: str = "You are a helpful LLM used for document structuring.",
) -> LLMClient:
    """Create a LangChain agent and wrap it in a LangChainLLMClient adapter.

    This preserves the existing get_agent(...) entrypoint while offering a
    backend-agnostic interface via LLMClient.
    """
    agent = get_agent(
        model=model,
        name=name,
        description=description,
        instruction=instruction,
    )
    return LangChainLLMClient(agent)


if __name__ == "__main__":
    try:
        agent = get_agent(
            model="gemini-2.5-flash",
            name="ahgc_demo_agent",
            description="Demo agent for AHGC LLM client",
            instruction="You answer briefly.",
        )
        reply = generate_with_agent(agent, "Say hi in one sentence.")
        print("Agent reply:", reply)
    except Exception as exc:  # noqa: BLE001
        print("[client demo] Unable to run demo:", exc)

