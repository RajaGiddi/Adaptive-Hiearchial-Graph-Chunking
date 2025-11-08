"""Generate retrieval query JSON files for each cleaned arXiv text document.

For every .txt file in an input directory (e.g. data/arxiv_texts/), this script
invokes an LLM (Gemini 2.5 flash by default) to synthesize 10 retrieval-style
queries plus a list of MUST-CONTAIN keyword stems for simple recall matching.

Output: one JSON file per input text, same base name, `.json` extension, e.g.
  a_characterization_of_strategy-proof_probabilistic_assignment_rules.json

Each JSON file is an array of objects:
[
  {
    "query": "What is the main problem addressed in this paper?",
    "must_contain": ["problem", "assignment", "mechanism", "strategy-proof", "probabilistic"]
  },
  ... (total 10 entries)
]

CLI:
  python scripts/generate_queries.py \
    --input_dir data/arxiv_texts \
    --output_dir data/queries \
    --model gemini-2.5-flash \
    --limit 5 --overwrite

Features / Behavior:
  - Truncates source text to --max_chars (default 12000) for prompt efficiency.
  - Simple keyword pre-extraction (frequency-based) fed into the prompt to guide LLM.
  - Robust JSON parsing: strips code fences; extracts first valid JSON object/array.
  - Fallback deterministic query generation if the LLM call or parsing fails.
  - Skips existing output unless --overwrite is set.
  - Loads .env (if present) to pick up GOOGLE_API_KEY.

Deterministic Fallback Strategy:
  - Extract top domain-ish tokens (lowercased) excluding a small stopword list.
  - Fill a templated list of 10 generic information-seeking queries referencing top tokens.

NOTE: This script depends on `ahgc.llm.client` utilities already present in repo.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List

# Ensure repository root is importable when invoked as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Lightweight .env loader (reuse concept from other scripts)
# ---------------------------------------------------------------------------
def _load_env_file_if_present(env_path: Path) -> None:
    try:
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_STOP = set(
    [
        "the","and","of","in","to","for","with","on","by","an","a","is","are","be","we","this","that","as","at","from","can","it","or","their","our","via","using","into","these","those","such","have","has","was","were","will","may","also","both","between","within","over","under","through","than","more","less"
    ]
)


def _basic_tokens(text: str) -> List[str]:
    text = text.lower()
    # Replace hyphens/underscores with space, punctuation to space
    text = re.sub(r"[-_]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if t and t not in _STOP and len(t) > 2]
    return toks


def _top_keywords(text: str, k: int = 25) -> List[str]:
    toks = _basic_tokens(text)
    freq = Counter(toks)
    return [w for w,_ in freq.most_common(k)]


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def _build_prompt(paper_text: str, filename: str, top_keywords: List[str]) -> str:
    kw_str = ", ".join(top_keywords[:25])
    truncated_info = f"Text length (truncated): {len(paper_text)} chars."
    example_block = (
        "Example output schema (strict JSON array):\n"\
        "[\n  {\n    \"query\": \"What is the main problem addressed in this paper?\",\n"\
        "    \"must_contain\": [\"problem\", \"assignment\", \"mechanism\", \"strategy-proof\", \"probabilistic\"]\n  },\n"\
        "  {\n    \"query\": \"What are probabilistic assignment rules in this context?\",\n"\
        "    \"must_contain\": [\"probabilistic\", \"assignment\", \"rule\", \"mechanism\", \"allocation\"]\n  }\n]"
    )
    instructions = (
        "You are generating 10 retrieval queries for the given academic paper.\n"
        "Return ONLY a JSON array of exactly 10 objects. Each object has: \n"
        "  - query: a concise natural-language question (<= 120 chars).\n"
        "  - must_contain: array of 4-7 lowercase keyword stems likely found in relevant chunks.\n"
        "Guidelines: diversify concepts (problem, methods, assumptions, results, implications, limitations).\n"
        "Use domain-specific terms where appropriate; prefer stems (e.g., 'probabilistic' not 'probabilistically').\n"
        "Do NOT add explanatory text outside JSON. Do NOT include code fences."
    )
    return (
        f"Filename: {filename}\n{truncated_info}\nTop candidate keywords: {kw_str}\n\n"
        f"{instructions}\n\n{example_block}\n\nPaper Text (truncated below):\n" + paper_text
    )


def _call_llm(prompt: str, model: str) -> str:
    try:
        from ahgc.llm.client import get_agent, generate_with_agent
        agent = get_agent(
            model=model,
            name="query_generator",
            description="Generates retrieval queries with must_contain keyword lists",
            instruction="Return only a JSON array (no Markdown fences).",
        )
        resp = generate_with_agent(agent, prompt)
        return resp
    except Exception as e:
        print(f"[warn] LLM call failed: {e}", file=sys.stderr)
        return ""


def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
    if not raw:
        return []
    s = raw.strip()
    # Strip code fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1].strip()
    # Extract first '[' ... ']' span
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    candidate = s[start : end + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            # keep only dict entries with required keys
            cleaned = []
            for obj in data:
                if not isinstance(obj, dict):
                    continue
                q = str(obj.get("query", "")).strip()
                mc = obj.get("must_contain")
                if q and isinstance(mc, list) and mc:
                    cleaned.append({"query": q, "must_contain": [str(x).lower() for x in mc if str(x).strip()]})
            return cleaned
    except Exception:
        return []
    return []


# ---------------------------------------------------------------------------
# Fallback deterministic queries
# ---------------------------------------------------------------------------
_FALLBACK_TEMPLATES = [
    "What primary problem does this paper address?",
    "How is the core method or approach defined?",
    "What assumptions are made about the entities or variables?",
    "Which theoretical results or propositions are central?",
    "How does the proposed method compare with prior work?",
    "What datasets, simulations, or examples illustrate the approach?",
    "How are efficiency or performance metrics characterized?",
    "What are key limitations or open challenges noted?",
    "What broader implications follow from the findings?",
    "How does the paper position itself in related literature?",
]


def _fallback_queries(keywords: List[str]) -> List[Dict[str, Any]]:
    # Use top ~35 keywords; group slices for must_contain lists.
    kw = keywords[:35] if keywords else []
    if not kw:
        kw = ["method","result","approach","model","analysis","data","performance"]
    out: List[Dict[str, Any]] = []
    stride = max(4, min(7, len(kw) // 10 or 4))
    for i, template in enumerate(_FALLBACK_TEMPLATES):
        slice_start = (i * stride) % len(kw)
        mc = kw[slice_start : slice_start + stride]
        out.append({"query": template, "must_contain": mc})
    return out


# ---------------------------------------------------------------------------
# Main per-file generation
# ---------------------------------------------------------------------------
def _generate_for_file(
    path: Path,
    model: str,
    overwrite: bool,
    max_chars: int,
    output_dir: Path,
    allow_fallback: bool,
) -> None:
    if path.name == "index.json":
        return
    out_path = output_dir / (path.stem + ".json")
    if out_path.exists() and not overwrite:
        return
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[warn] failed to read {path.name}: {e}", file=sys.stderr)
        return
    truncated = raw[:max_chars]
    keywords = _top_keywords(truncated, k=40)
    prompt = _build_prompt(truncated, path.name, keywords)
    resp = _call_llm(prompt, model=model)
    parsed = _parse_json_array(resp)
    if len(parsed) != 10:
        if allow_fallback:
            print(
                f"[info] LLM output unusable or wrong length for {path.name}; using fallback",
                file=sys.stderr,
            )
            parsed = _fallback_queries(keywords)
        else:
            print(
                f"[error] LLM output unusable or wrong length for {path.name}; skipping (use --allow_fallback to synthesize)",
                file=sys.stderr,
            )
            return
    # Write
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"generated queries: {path.name} → {out_path.name} (queries={len(parsed)})")
    except Exception as e:
        print(f"[warn] failed to write {out_path}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate retrieval query JSON files for arXiv texts")
    parser.add_argument("--input_dir", required=True, help="Directory of cleaned .txt files")
    parser.add_argument("--output_dir", required=True, help="Directory to write query .json files")
    parser.add_argument("--model", default="gemini-2.5-flash", help="LLM model name (default: gemini-2.5-flash)")
    parser.add_argument("--limit", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .json files")
    parser.add_argument("--max_chars", type=int, default=12000, help="Max chars of source text to feed into prompt")
    parser.add_argument(
        "--allow_fallback",
        action="store_true",
        help="If set, generate deterministic fallback queries when LLM fails. By default, strict: skip file on failure.",
    )
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load .env for API keys
    _load_env_file_if_present(Path(__file__).resolve().parents[1] / ".env")

    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix == ".txt"])
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    for f in files:
        _generate_for_file(
            f,
            model=args.model,
            overwrite=args.overwrite,
            max_chars=args.max_chars,
            output_dir=output_dir,
            allow_fallback=bool(args.allow_fallback),
        )


if __name__ == "__main__":
    main()
