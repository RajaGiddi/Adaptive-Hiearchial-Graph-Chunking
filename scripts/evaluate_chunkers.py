"""Evaluate multiple chunking strategies on a single input document.

This script runs a set of chunkers (including AHGC) and prints a unified report.
Optionally writes JSONL where each line is the standardized result dict for one
method.

Standardized result shape per method:
{
  "method": str,
  "num_chunks": int,
  "avg_chunk_len": float,  # in characters
  "max_chunk_len": int,
  "min_chunk_len": int,
  "chunks": list[str],     # optional, for debugging
  # AHGC-only extras
  "num_sections": int,
  "num_graph_nodes": int,
  "num_graph_edges": int,
}

Notes / TODOs:
- Real tokenization (e.g., tiktoken) should replace char counts where noted.
- Semantic chunking is a placeholder and uses a single LLM call.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from statistics import pstdev

# Ensure repository root is importable when invoked as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------- helpers & utilities ----------------------------

def _normalize_chunks(chunks: Iterable[str]) -> List[str]:
    out: List[str] = []
    for c in chunks:
        s = (c or "").strip()
        if s:
            out.append(s)
    return out


def _stats(chunks: List[str]) -> Tuple[int, float, int, int]:
    if not chunks:
        return 0, 0.0, 0, 0
    lens = [len(c) for c in chunks]
    n = len(lens)
    return n, sum(lens) / n, max(lens), min(lens)


def _extra_metrics(chunks: List[str], source_len: int) -> Tuple[float, int, float]:
    """Compute additional metrics for a chunk list.

    Returns:
    - std_chunk_len: population standard deviation of chunk lengths
    - total_chars: total characters across all chunks
    - compression_ratio: len(source_text)/total_chars (0.0 if total_chars==0)
    """
    if not chunks:
        return 0.0, 0, 0.0
    lens = [len(c) for c in chunks]
    std = pstdev(lens) if len(lens) > 1 else 0.0
    total = sum(lens)
    ratio = (source_len / total) if total > 0 else 0.0
    return std, total, ratio


def _split_sentences(text: str) -> List[str]:
    # naive sentence splitting on ., ?, ! (keep punctuation)
    parts = re.split(r"([\.\?\!])", text)
    # re-attach delimiters
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        sent = parts[i]
        if i + 1 < len(parts):
            sent += parts[i + 1]
        if sent.strip():
            sentences.append(sent.strip())
    return sentences


def _sent_tokenize_smart(text: str) -> List[str]:
    """Tokenize sentences using NLTK or spaCy when available, else fallback.

    Preference:
    - nltk.sent_tokenize (downloads 'punkt' if missing)
    - spaCy (load en_core_web_sm if present, else use a blank pipeline with sentencizer)
    - naive regex splitter as last resort
    """
    # Try NLTK first
    try:
        from nltk.tokenize import sent_tokenize  # type: ignore

        try:
            sents = sent_tokenize(text)
        except LookupError:
            # Attempt to fetch the punkt model silently
            try:
                import nltk  # type: ignore

                nltk.download("punkt", quiet=True)
            except Exception:
                pass
            sents = sent_tokenize(text)
        return [s.strip() for s in sents if s and s.strip()]
    except Exception:
        # Try spaCy next
        try:
            import spacy  # type: ignore

            try:
                nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])  # type: ignore[arg-type]
            except Exception:
                nlp = spacy.blank("en")  # type: ignore
                if "sentencizer" not in nlp.pipe_names:  # type: ignore[attr-defined]
                    nlp.add_pipe("sentencizer")  # type: ignore[attr-defined]
            doc = nlp(text)
            sents = [s.text.strip() for s in getattr(doc, "sents", []) if s.text.strip()]
            if sents:
                return sents
        except Exception:
            pass
    # Fallback naive splitter
    return _split_sentences(text)


def _group_sentences(sentences: List[str], max_per_chunk: int = 3) -> List[str]:
    chunks: List[str] = []
    for i in range(0, len(sentences), max_per_chunk):
        group = sentences[i : i + max_per_chunk]
        if group:
            chunks.append(" ".join(group))
    return chunks


def _split_paragraphs(text: str) -> List[str]:
    # reuse the idea of paragraph segmentation: double newline
    return [p for p in text.split("\n\n")]


def _fixed_size(text: str, size: int = 800) -> List[str]:
    text = text.strip()
    return [text[i : i + size] for i in range(0, len(text), size)]


def _sliding_window(text: str, size: int = 800, stride: int = 400) -> List[str]:
    text = text.strip()
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        if i + size >= len(text):
            break
        i += stride
    return chunks


def _recursive_chunks(text: str, para_max: int = 1200, fixed_fallback: int = 800) -> List[str]:
    out: List[str] = []
    for p in _split_paragraphs(text):
        if len(p) <= para_max:
            out.append(p)
            continue
        # sentence-level split
        sents = _split_sentences(p)
        sent_groups = _group_sentences(sents, max_per_chunk=3)
        for s in sent_groups:
            if len(s) <= para_max:
                out.append(s)
            else:
                out.extend(_fixed_size(s, size=fixed_fallback))
    return out


def _delimiter_split(text: str, delimiter: str = "---") -> List[str]:
    if delimiter in text:
        return text.split(delimiter)
    if "###" in text:
        return text.split("###")
    # fallback to newline-based paragraphs
    return _split_paragraphs(text)


def _format_aware_chunks(text: str) -> List[str]:
    # Keep markdown headings and code fences as intact blocks
    lines = text.splitlines()
    chunks: List[str] = []
    buf: List[str] = []
    in_code = False
    for ln in lines:
        if ln.strip().startswith("```"):
            if in_code:
                # closing fence
                buf.append(ln)
                chunks.append("\n".join(buf))
                buf = []
                in_code = False
            else:
                # starting fence: flush current buf as a chunk
                if buf:
                    chunks.append("\n".join(buf))
                    buf = []
                in_code = True
                buf.append(ln)
            continue
        if in_code:
            buf.append(ln)
            continue
        if re.match(r"^#+\s+", ln):  # heading
            if buf:
                chunks.append("\n".join(buf))
                buf = []
            buf.append(ln)
        else:
            buf.append(ln)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


# ------------------------------- chunker methods -----------------------------

def run_ahgc(text: str, model: str = "gemini-2.5-flash", **_: Any) -> Dict[str, Any]:
    """Run the AHGC pipeline and return standardized metrics.

    Uses existing modules: segment -> summarize -> induce_hierarchy -> materialize
    -> build_final_chunks -> build_graph_from_sections.
    """
    from ahgc.ingestion.segmentation import segment_document
    from ahgc.llm.summarizer import summarize_segments
    from ahgc.llm.grouper import induce_hierarchy
    from ahgc.chunking.materialize import materialize_sections
    from ahgc.chunking.rechunk import build_final_chunks
    from ahgc.graph.builder import build_graph_from_sections, GraphBuildConfig

    segments = segment_document(text)
    summarized = summarize_segments(segments, model=model, batch_size=5)
    hierarchy = induce_hierarchy(summarized_segments=summarized, model=model)
    sections = materialize_sections(hierarchy, segments)
    chunks = build_final_chunks(doc_id="doc", sections=sections, max_tokens=200, overlap=30)
    graph = build_graph_from_sections(
        doc_id="doc", sections=sections, chunks=chunks, config=GraphBuildConfig()
    )

    # Extract chunk texts
    chunk_texts = _normalize_chunks([str(c.get("text", "")) for c in chunks])
    n, avg_len, max_len, min_len = _stats(chunk_texts)
    std_len, total_chars, compression_ratio = _extra_metrics(chunk_texts, len(text))

    # Graph counts
    try:
        num_nodes = int(graph.number_of_nodes())  # type: ignore[attr-defined]
        num_edges = int(graph.number_of_edges())  # type: ignore[attr-defined]
    except Exception:
        try:
            num_nodes = len(graph.get("nodes", {}))  # type: ignore[call-arg]
            num_edges = len(graph.get("edges", []))  # type: ignore[call-arg]
        except Exception:
            num_nodes = 0
            num_edges = 0

    return {
        "method": "ahgc",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": max_len,
        "min_chunk_len": min_len,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": chunk_texts,
        "num_sections": len(sections),
        "num_graph_nodes": num_nodes,
        "num_graph_edges": num_edges,
    }


def run_fixed(text: str, size: int = 800, **_: Any) -> Dict[str, Any]:
    parts = _normalize_chunks(_fixed_size(text, size=size))
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "fixed",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_sliding(text: str, size: int = 800, stride: int = 400, **_: Any) -> Dict[str, Any]:
    parts = _normalize_chunks(_sliding_window(text, size=size, stride=stride))
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "sliding",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_sentence(text: str, max_per_chunk: int = 3, **_: Any) -> Dict[str, Any]:
    """Sentence-level chunking using NLTK/spaCy with 2–3 sentences per chunk.

    Defaults to grouping up to 3 sentences; adjust max_per_chunk to 2 for
    smaller chunks. Falls back to a naive splitter if libraries are missing.
    """
    sents = _sent_tokenize_smart(text)
    parts = _normalize_chunks(_group_sentences(sents, max_per_chunk=max_per_chunk))
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "sentence",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_paragraph(text: str, **_: Any) -> Dict[str, Any]:
    parts = _normalize_chunks(_split_paragraphs(text))
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "paragraph",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_recursive(text: str, para_max: int = 800, fixed_fallback: int = 800, **_: Any) -> Dict[str, Any]:
    """Recursive/hierarchical split:

    - Start with paragraph chunks.
    - For any paragraph > para_max (default 800 chars), split by sentences and
      group a few per chunk.
    - If any resulting chunk is still > para_max, fall back to fixed-size
      splitting with size=fixed_fallback.
    """
    refined: List[str] = []
    for p in _split_paragraphs(text):
        if len(p) <= para_max:
            refined.append(p)
            continue
        sents = _split_sentences(p)
        groups = _group_sentences(sents, max_per_chunk=3)
        for g in groups:
            if len(g) <= para_max:
                refined.append(g)
            else:
                refined.extend(_fixed_size(g, size=fixed_fallback))
    parts = _normalize_chunks(refined)
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "recursive",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_semantic(text: str, model: str = "gemini-2.5-flash", **_: Any) -> Dict[str, Any]:
    # Placeholder semantic splitting using one LLM call returning JSON
    prompt = (
        "Split the following text into semantically coherent sections. Return JSON: "
        "{'chunks': ['...']}\n\n" + text
    )
    try:
        from ahgc.llm.client import get_agent, generate_with_agent

        agent = get_agent(
            model=model,
            name="ahgc_semantic_chunker",
            description="Splits text into semantically coherent chunks",
            instruction="Return only JSON with a 'chunks' array of strings.",
        )
        resp = generate_with_agent(agent, prompt)
        # try parse JSON (strip code fences if present)
        s = resp.strip()
        if s.startswith("```"):
            parts = s.split("```")
            if len(parts) >= 3:
                s = parts[1].strip()
        data = None
        try:
            data = json.loads(s)
        except Exception:
            # attempt to extract first JSON object
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                data = json.loads(s[start : end + 1])
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            parts = _normalize_chunks([str(x) for x in data["chunks"]])
        else:
            raise ValueError("invalid JSON structure")
    except Exception:
        # Fallback to a sentence-level grouping (avoid paragraph-only fallback)
        sents = _split_sentences(text)
        parts = _normalize_chunks(_group_sentences(sents, max_per_chunk=3))

    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "semantic",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_delimiter(text: str, delimiter: str = "---", **_: Any) -> Dict[str, Any]:
    parts = _normalize_chunks(_delimiter_split(text, delimiter=delimiter))
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "delimiter",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_token_limit(text: str, limit: int = 800, **_: Any) -> Dict[str, Any]:
    """Token-limit chunking using a real tokenizer when available.

    Preference order:
    - tiktoken (enc='cl100k_base') for encoding/decoding and fixed-size slicing
    - langchain TokenTextSplitter as a fallback splitter
    - character-based fallback (with a stderr warning)

    Returns both character stats and token-based stats per chunk.
    """
    chunks: List[str] = []
    token_lens: List[int] = []
    tokenizer_name = ""

    # Try tiktoken first
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        tokenizer_name = "tiktoken:cl100k_base"
        tokens = enc.encode(text)
        for i in range(0, len(tokens), int(limit)):
            toks = tokens[i : i + int(limit)]
            s = enc.decode(toks)
            s = s.strip()
            if s:
                chunks.append(s)
                token_lens.append(len(toks))
    except Exception:
        # Fallback: try LangChain's TokenTextSplitter (old/new import paths)
        try:
            try:
                from langchain.text_splitter import TokenTextSplitter  # type: ignore
            except Exception:  # pragma: no cover - optional dependency path
                from langchain_text_splitters import TokenTextSplitter  # type: ignore

            splitter = TokenTextSplitter(chunk_size=int(limit), chunk_overlap=0)
            chunks = _normalize_chunks(splitter.split_text(text))
            tokenizer_name = "langchain:TokenTextSplitter"
            # Try to compute token lengths with tiktoken if available; else estimate
            try:
                import tiktoken  # type: ignore

                enc = tiktoken.get_encoding("cl100k_base")
                token_lens = [len(enc.encode(c)) for c in chunks]
            except Exception:
                # Heuristic: ~4 chars per token if tokenizer missing
                token_lens = [max(1, len(c) // 4) for c in chunks]
        except Exception:
            # Last resort: character-based split with warning
            print(
                "[warn] falling back to character-based token_limit splitting; install 'tiktoken'",
                file=sys.stderr,
            )
            chunks = _normalize_chunks(_fixed_size(text, size=limit))
            token_lens = [max(1, len(c) // 4) for c in chunks]  # heuristic

    # Character stats
    n, avg_chars, max_chars, min_chars = _stats(chunks)
    std_len, total_chars, compression_ratio = _extra_metrics(chunks, len(text))
    # Token stats
    if token_lens:
        avg_tokens = sum(token_lens) / len(token_lens)
        max_tokens = max(token_lens)
        min_tokens = min(token_lens)
    else:
        avg_tokens = 0.0
        max_tokens = 0
        min_tokens = 0

    return {
        "method": "token_limit",
        "num_chunks": n,
        "avg_chunk_len": avg_chars,
        "std_chunk_len": std_len,
        "max_chunk_len": max_chars,
        "min_chunk_len": min_chars,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "avg_chunk_tokens": avg_tokens,
        "max_chunk_tokens": max_tokens,
        "min_chunk_tokens": min_tokens,
        "tokenizer": tokenizer_name,
        "chunks": chunks,
    }


def run_format_aware(text: str, **_: Any) -> Dict[str, Any]:
    """Format-aware chunking for Markdown.

    Rules:
    - Detect headings via regex: ^#{1,6}\\s+
    - Treat fenced code blocks (``` ... ```) as atomic chunks.
    - Maintain a heading together with its following block (content/code).
    """

    lines = text.splitlines()
    chunks: List[str] = []

    heading_re = re.compile(r"^#{1,6}\s+")
    in_code = False
    code_buf: List[str] = []
    section_buf: List[str] = []  # accumulates heading + following block(s)

    def flush_section() -> None:
        nonlocal section_buf
        if section_buf:
            chunk = "\n".join(section_buf).strip()
            if chunk:
                chunks.append(chunk)
            section_buf = []

    def flush_code_atomic() -> None:
        nonlocal code_buf
        if code_buf:
            chunk = "\n".join(code_buf).strip()
            if chunk:
                chunks.append(chunk)
            code_buf = []

    for ln in lines:
        stripped = ln.strip()
        # Toggle fenced code blocks
        if stripped.startswith("```"):
            if in_code:
                # closing fence
                if section_buf:
                    # include code block inside the current heading section
                    section_buf.append(ln)
                else:
                    code_buf.append(ln)
                    flush_code_atomic()
                in_code = False
            else:
                # opening fence
                if section_buf:
                    # keep it within section (do not split)
                    section_buf.append(ln)
                else:
                    # start atomic code chunk
                    flush_code_atomic()  # just in case
                    code_buf.append(ln)
                in_code = True
            continue

        if in_code:
            # Inside code block
            if section_buf:
                section_buf.append(ln)
            else:
                code_buf.append(ln)
            continue

        # Headings detected only when not in code
        if heading_re.match(ln):
            # new section: flush previous section first
            flush_section()
            section_buf.append(ln)
            continue

        # Normal line: attach to current section if present, else accumulate
        if section_buf:
            section_buf.append(ln)
        else:
            # No heading context -> treat contiguous text until next heading as one chunk
            section_buf.append(ln)

    # Final flushes
    if in_code:
        # Unbalanced fence: treat accumulated as atomic
        if section_buf:
            section_buf.extend(code_buf)
            code_buf = []
        else:
            flush_code_atomic()
        in_code = False

    flush_section()

    parts = _normalize_chunks(chunks)
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "format_aware",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


def run_hybrid(text: str, long_threshold: int = 800, **_: Any) -> Dict[str, Any]:
    hybrid: List[str] = []
    for p in _split_paragraphs(text):
        if len(p) <= long_threshold:
            hybrid.append(p)
        else:
            sents = _split_sentences(p)
            hybrid.extend(_group_sentences(sents, max_per_chunk=3))
    parts = _normalize_chunks(hybrid)
    n, avg_len, mx, mn = _stats(parts)
    std_len, total_chars, compression_ratio = _extra_metrics(parts, len(text))
    return {
        "method": "hybrid",
        "num_chunks": n,
        "avg_chunk_len": avg_len,
        "std_chunk_len": std_len,
        "max_chunk_len": mx,
        "min_chunk_len": mn,
        "total_chars": total_chars,
        "compression_ratio": compression_ratio,
        "chunks": parts,
    }


# Registry of chunkers
CHUNKERS = {
    "ahgc": run_ahgc,
    "fixed": run_fixed,
    "sliding": run_sliding,
    "sentence": run_sentence,
    "paragraph": run_paragraph,
    "recursive": run_recursive,
    "semantic": run_semantic,
    "delimiter": run_delimiter,
    "token_limit": run_token_limit,
    "format_aware": run_format_aware,
    "hybrid": run_hybrid,
}


def _print_table(results: List[Dict[str, Any]]) -> None:
    print("method, num_chunks, avg_len, std_len, max_len, min_len, total_chars, compression_ratio")
    for r in results:
        print(
            f"{r['method']}, {r.get('num_chunks', 0)}, "
            f"{round(float(r.get('avg_chunk_len', 0.0)), 2)}, "
            f"{round(float(r.get('std_chunk_len', 0.0)), 2)}, "
            f"{r.get('max_chunk_len', 0)}, {r.get('min_chunk_len', 0)}, "
            f"{r.get('total_chars', 0)}, {round(float(r.get('compression_ratio', 0.0)), 4)}"
        )


def _load_env_file_if_present(env_path: Path) -> None:
    """Lightweight .env loader to populate GOOGLE_API_KEY if present.

    Supports lines KEY=VALUE with optional quotes. Skips comments and blanks.
    Does not override existing environment variables.
    """
    try:
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            key, val = s.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # Best-effort; ignore parsing errors
        pass


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple chunkers on a document")
    parser.add_argument("--input", required=True, help="Path to .txt / .md file")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="LLM model name for AHGC & semantic methods (default: gemini-2.5-flash)",
    )
    parser.add_argument("--output", help="Optional JSONL output path (one line per method)")
    parser.add_argument(
        "--methods",
        help=(
            "Comma-separated list of methods to run. "
            "Defaults to all: " + ",".join(CHUNKERS.keys())
        ),
    )
    args = parser.parse_args(argv)

    path = args.input
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Load .env if present to get GOOGLE_API_KEY for LLM-backed methods
    _load_env_file_if_present(_REPO_ROOT / ".env")

    text = Path(path).read_text(encoding="utf-8")

    # Determine methods to run
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        methods = list(CHUNKERS.keys())

    # Execute chunkers
    results: List[Dict[str, Any]] = []
    for m in methods:
        fn = CHUNKERS.get(m)
        if not fn:
            print(f"[warn] Unknown method '{m}', skipping", file=sys.stderr)
            continue
        try:
            kwargs = {"model": args.model}
            res = fn(text, **kwargs)
            res["method"] = m  # enforce method key
            results.append(res)
        except Exception as e:  # pragma: no cover - robustness for ad-hoc runs
            print(f"[error] Method '{m}' failed: {e!r}", file=sys.stderr)

    # Print table to stdout
    _print_table(results)

    # Optional JSONL output
    if args.output:
        out_path = Path(args.output)
        # Ensure parent directory exists (e.g., when using paths like tmp/chunks.jsonl)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best effort; will error on open if still invalid
            pass
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
