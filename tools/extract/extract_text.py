"""
Extract per-chapter text from the PRML PDF, with page markers, for downstream
chapter-writing agents to cite book pages against.

Run: uv run --with pymupdf python tools/extract/extract_text.py
Reads: the PDF named by $PRML_PDF (default: ../Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, one level above the repo)
Writes: tools/extract/out/front.txt, ch01.txt .. ch14.txt, appendix.txt
"""

import sys
from pathlib import Path

import fitz

from _pdf_toc import (
    NUM_CHAPTERS,
    PDF_PATH,
    derive_offset,
    parse_toc_appendix_a_start,
    parse_toc_chapter_starts,
)

OUT_DIR = Path(__file__).parent / "out"


def main() -> None:
    doc = fitz.open(PDF_PATH)
    print(f"PDF has {doc.page_count} physical pages")

    chapter_starts = parse_toc_chapter_starts(doc)
    appendix_a_book_start = parse_toc_appendix_a_start(doc)
    print("Chapter start book pages (from TOC):")
    for n in sorted(chapter_starts):
        print(f"  ch{n:02d}: book page {chapter_starts[n]}")
    print(f"  appendix A: book page {appendix_a_book_start}")

    offset = derive_offset(doc, chapter_starts)
    print(f"Derived pdf_page = book_page + {offset}")

    # 0-based pdf index for the first page of each chapter / appendix front matter.
    pdf_start = {n: chapter_starts[n] - 1 + offset for n in chapter_starts}
    appendix_pdf_start = appendix_a_book_start - 1 + offset

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def write_range(name: str, start_idx: int, end_idx_inclusive: int) -> None:
        parts = []
        for idx in range(start_idx, end_idx_inclusive + 1):
            page_num = idx + 1  # 1-based physical page index, per spec
            parts.append(f"\n===== page {page_num:03d} =====\n")
            parts.append(doc[idx].get_text())
        text = "".join(parts)
        out_path = OUT_DIR / name
        out_path.write_text(text, encoding="utf-8")
        print(
            f"wrote {out_path} ({end_idx_inclusive - start_idx + 1} pages, {len(text)} chars)"
        )

    write_range("front.txt", 0, pdf_start[1] - 1)

    for n in range(1, NUM_CHAPTERS + 1):
        start = pdf_start[n]
        end = (pdf_start[n + 1] - 1) if n < NUM_CHAPTERS else (appendix_pdf_start - 1)
        write_range(f"ch{n:02d}.txt", start, end)

    write_range("appendix.txt", appendix_pdf_start, doc.page_count - 1)


if __name__ == "__main__":
    sys.exit(main())
