"""
Shared table-of-contents parsing and physical/book page mapping for the PRML PDF.

Used by extract_text.py, extract_equations.py, and extract_figures.py so the
book-page <-> pdf-page offset is derived once, the same way, everywhere.
"""

import os
import re
from pathlib import Path

import fitz

PDF_FILENAME = "Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"

# The PDF is deliberately outside the repo, because PRML is free for personal use but not
# redistributable. It sits one level above the repo root by default; PRML_PDF overrides.
PDF_PATH = os.environ.get(
    "PRML_PDF",
    str(Path(__file__).resolve().parents[2].parent / PDF_FILENAME),
)

NUM_CHAPTERS = 14

# The table of contents lives on PDF pages 8-24 (1-based), i.e. 0-based indices 7-23.
TOC_PAGE_RANGE = range(7, 24)


def _toc_lines(doc: fitz.Document) -> list[str]:
    lines = []
    for i in TOC_PAGE_RANGE:
        for ln in doc[i].get_text().split("\n"):
            lines.append(ln.strip())
    return lines


def parse_toc_chapter_starts(doc: fitz.Document) -> dict[int, int]:
    """Return {chapter_number: book_start_page} parsed from the table of contents.

    PyMuPDF's line ordering for the TOC varies: single-digit chapters get their
    number, title, and page on separate lines, but two-digit chapters (10-14) get
    the number and title on one combined line. The regex below tolerates either
    layout by anchoring on the chapter's own "N.1" first-subsection marker, which
    is the one token that reliably follows the chapter's start page number.
    """
    full = "\n".join(_toc_lines(doc))

    starts: dict[int, int] = {}
    for num in range(1, NUM_CHAPTERS + 1):
        pattern = re.compile(
            rf"(?m)^{num}(?: .+)?$\n(?:.+\n)*?(\d{{1,4}})\n{num}\.1(?: .*)?$"
        )
        m = pattern.search(full)
        if not m:
            raise RuntimeError(f"could not locate chapter {num} in table of contents")
        starts[num] = int(m.group(1))
    return starts


def parse_toc_appendix_a_start(doc: fitz.Document) -> int:
    lines = _toc_lines(doc)
    for idx, ln in enumerate(lines):
        if ln == "Appendix A":
            for j in range(idx + 1, min(idx + 4, len(lines))):
                if re.fullmatch(r"\d{1,4}", lines[j]):
                    return int(lines[j])
    raise RuntimeError("could not locate Appendix A in table of contents")


def find_chapter1_pdf_index(
    doc: fitz.Document, search_from: int = 15, search_to: int = 60
) -> int:
    """Locate chapter 1's opening page by its literal heading, to derive the
    physical-page/book-page offset instead of assuming it."""
    for idx in range(search_from, search_to):
        lines = [ln.strip() for ln in doc[idx].get_text().split("\n") if ln.strip()]
        if len(lines) >= 2 and lines[0] == "1" and lines[1] == "Introduction":
            return idx
    raise RuntimeError("could not locate chapter 1 opening page to derive page offset")


def derive_offset(doc: fitz.Document, chapter_starts: dict[int, int]) -> int:
    chapter1_pdf_index = find_chapter1_pdf_index(doc)
    return chapter1_pdf_index - (chapter_starts[1] - 1)
