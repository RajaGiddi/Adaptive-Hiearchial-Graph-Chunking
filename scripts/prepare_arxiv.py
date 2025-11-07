"""Prepare arXiv PDFs for chunker evaluation by converting to cleaned .txt.

Usage:
  python scripts/prepare_arxiv.py \
      --input_dir data/arxiv_pdfs \
      --output_dir data/arxiv_texts \
      --limit 10 --overwrite

Notes:
- Prefers pdfminer.six for text extraction; falls back to `pdftotext` if available.
- Cleans and normalizes text for better downstream chunking.
- Keeps the script standalone (no dependencies on AHGC code).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# ----------------------------- PDF extraction -----------------------------

def _extract_with_pdfminer(pdf_path: Path) -> Optional[str]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        return extract_text(str(pdf_path))
    except Exception as e:
        print(f"[warn] pdfminer failed for {pdf_path.name}: {e}", file=sys.stderr)
        return None


def _extract_with_pdftotext(pdf_path: Path) -> Optional[str]:
    """Use the `pdftotext` CLI if installed (poppler-utils)."""
    try:
        # -enc UTF-8 ensures proper encoding; '-' writes to stdout
        proc = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(pdf_path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.decode("utf-8", errors="replace")
    except FileNotFoundError:
        print("[warn] pdftotext not found in PATH", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        print(f"[warn] pdftotext failed for {pdf_path.name}: {e}", file=sys.stderr)
        return None


def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """Best-effort extraction: pdfminer first, then pdftotext."""
    txt = _extract_with_pdfminer(pdf_path)
    if txt and txt.strip():
        return txt
    return _extract_with_pdftotext(pdf_path)


# ----------------------------- Text normalization -----------------------------

_SECTION_RE = re.compile(r"\\section\*?\{([^}]*)\}")
_SUBSECTION_RE = re.compile(r"\\subsection\*?\{([^}]*)\}")
_SUBSUBSECTION_RE = re.compile(r"\\subsubsection\*?\{([^}]*)\}")


def _convert_latex_headings(text: str) -> str:
    text = _SECTION_RE.sub(lambda m: f"### {m.group(1).strip()}", text)
    text = _SUBSECTION_RE.sub(lambda m: f"#### {m.group(1).strip()}", text)
    text = _SUBSUBSECTION_RE.sub(lambda m: f"##### {m.group(1).strip()}", text)
    return text


_NUM_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+?)\s*$")


def _ensure_numeric_headings_on_own_lines(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        m = _NUM_HEADING_RE.match(ln)
        if m:
            # Collapse whitespace in heading title
            title = re.sub(r"\s+", " ", m.group(2)).strip()
            out.append(f"{m.group(1)} {title}")
        else:
            out.append(ln)
    return "\n".join(out)


_REFS_RE = re.compile(r"^\s*(References|Bibliography)\b.*$", re.IGNORECASE | re.MULTILINE)


def _remove_references_block(text: str) -> str:
    m = _REFS_RE.search(text)
    if not m:
        return text
    # Truncate everything from the references heading to end
    return text[: m.start()].rstrip() + "\n"


def _normalize_newlines(text: str) -> str:
    # Normalize Windows/Mac line endings first
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of blank lines into a single blank line
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    # Trim trailing spaces per line
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return text


def _strip_non_ascii_but_keep_greek(text: str) -> str:
    out_chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if 0x20 <= code <= 0x7E or ch in "\n\t":
            out_chars.append(ch)
        elif 0x0370 <= code <= 0x03FF:  # Greek & Coptic
            out_chars.append(ch)
        else:
            # Replace other non-ASCII with a space to avoid word sticking
            out_chars.append(" ")
    s = "".join(out_chars)
    # Normalize multiple spaces
    s = re.sub(r"[ \t]+", " ", s)
    return s


def clean_text(raw: str) -> str:
    # Convert LaTeX headings first
    txt = _convert_latex_headings(raw)
    # Ensure numeric headings are preserved as their own lines
    txt = _ensure_numeric_headings_on_own_lines(txt)
    # Remove references/bibliography section
    txt = _remove_references_block(txt)
    # Normalize newlines and whitespace
    txt = _normalize_newlines(txt)
    # Strip most non-ASCII but keep Greek letters
    txt = _strip_non_ascii_but_keep_greek(txt)
    # Final pass: collapse excessive blank lines again (post-clean)
    txt = re.sub(r"\n\s*\n+", "\n\n", txt).strip() + "\n"
    return txt


# --------------------------------- Main ----------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare arXiv PDFs into cleaned .txt files")
    parser.add_argument("--input_dir", required=True, help="Directory containing arXiv PDFs")
    parser.add_argument("--output_dir", required=True, help="Directory to write cleaned .txt files")
    parser.add_argument("--limit", type=int, default=0, help="Number of PDFs to process (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    args = parser.parse_args(argv)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional progress bar
    try:
        from tqdm import tqdm  # type: ignore

        progress = tqdm
    except Exception:
        def progress(x, **kwargs):  # type: ignore
            return x

    pdfs = sorted([p for p in in_dir.iterdir() if p.suffix.lower() == ".pdf"])
    if args.limit and args.limit > 0:
        pdfs = pdfs[: args.limit]

    processed = 0
    for pdf_path in progress(pdfs, desc="Preparing arXiv PDFs"):
        base = pdf_path.stem
        out_path = out_dir / f"{base}.txt"
        if out_path.exists() and not args.overwrite:
            continue
        try:
            raw = extract_text_from_pdf(pdf_path)
            if not raw or not raw.strip():
                print(f"[warn] empty extraction, skipping: {pdf_path.name}", file=sys.stderr)
                continue
            cleaned = clean_text(raw)
            # Sanity check: skip too-short outputs
            if len(cleaned) < 2000:
                print(f"[warn] too short after cleaning (<2000 chars), skipping: {pdf_path.name}", file=sys.stderr)
                continue
            out_path.write_text(cleaned, encoding="utf-8")
            processed += 1
            print(
                f"processed: {pdf_path.name} → {out_path.name} (chars={len(cleaned):,})"
            )
        except Exception as e:
            print(f"[warn] failed to process {pdf_path.name}: {e}", file=sys.stderr)
            continue

    if processed == 0:
        print("[info] No files processed.")


if __name__ == "__main__":
    main()
