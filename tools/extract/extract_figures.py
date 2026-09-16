"""
Build the committed figure-number manifest (numbers only, never caption text,
since Bishop's captions are copyrighted prose and this repo authors its own
descriptions).

Run: uv run --with pymupdf python tools/extract/extract_figures.py
Reads: the PDF named by $PRML_PDF (default: ../Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, one level above the repo)
Writes: tools/extract/figures.json (committed)
"""

import json
import re
import sys
from datetime import datetime, timezone

import fitz

from _pdf_toc import PDF_PATH, derive_offset, parse_toc_chapter_starts

OUT_PATH = __file__.rsplit("/", 1)[0] + "/figures.json"

CAPTION_RE = re.compile(r"^Figure\s+(\d{1,2})\.(\d{1,3})\b")

# A caption's own first line starts near the page's outer edge (~30-70pt on a
# 504pt page, alternating with odd/even mirrored margins). An inline mention
# like "Figure 3.11." as the first word of a body sentence starts at the
# normal body margin (~121pt) instead, so the same left-margin gap that
# separates equation labels from citations separates captions from mentions.
CAPTION_LEFT_MAX = 100


def find_figures(doc: fitz.Document, offset: int) -> tuple[list[dict], int]:
    figures = []
    rejected = 0

    for page_index in range(doc.page_count):
        page_dict = doc[page_index].get_text("dict")
        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                match = CAPTION_RE.match(line_text)
                if not match:
                    continue
                if line["bbox"][0] < CAPTION_LEFT_MAX:
                    chapter = int(match.group(1))
                    fig_id = f"{chapter}.{int(match.group(2))}"
                    pdf_page = page_index + 1
                    figures.append(
                        {
                            "id": fig_id,
                            "chapter": chapter,
                            "pdfPage": pdf_page,
                            "bookPage": pdf_page - offset,
                        }
                    )
                else:
                    rejected += 1

    return figures, rejected


def sort_key(fig: dict) -> tuple[int, float]:
    sub = fig["id"].split(".", 1)[1]
    return (fig["chapter"], int(sub))


def dedupe_keeping_first_occurrence(figures: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    dropped = []
    for fig in sorted(figures, key=lambda f: f["pdfPage"]):
        if fig["id"] in seen:
            dropped.append(fig)
            continue
        seen[fig["id"]] = fig
    if dropped:
        print(f"Dropped {len(dropped)} duplicate id(s) (kept earliest page):")
        for fig in dropped:
            print(
                f"  {fig['id']} at pdfPage {fig['pdfPage']} (kept pdfPage {seen[fig['id']]['pdfPage']})"
            )
    return list(seen.values())


def report_counts(figures: list[dict]) -> None:
    by_chapter: dict[int, list[int]] = {}
    for fig in figures:
        sub = int(fig["id"].split(".", 1)[1])
        by_chapter.setdefault(fig["chapter"], []).append(sub)
    for chapter in sorted(by_chapter):
        subs = sorted(by_chapter[chapter])
        gaps = []
        expected = 1
        for s in subs:
            if s != expected:
                gaps.append((expected, s - 1))
            expected = s + 1
        note = f"gaps: {gaps}" if gaps else "contiguous"
        print(
            f"  chapter {chapter}: {len(subs)} figures, range {subs[0]}-{subs[-1]} ({note})"
        )


def main() -> None:
    doc = fitz.open(PDF_PATH)
    chapter_starts = parse_toc_chapter_starts(doc)
    offset = derive_offset(doc, chapter_starts)
    print(f"Using pdf_page = book_page + {offset}")

    figures, rejected = find_figures(doc, offset)
    print(f"Found {len(figures)} figure captions, rejected {rejected} inline mentions")

    figures = dedupe_keeping_first_occurrence(figures)
    figures.sort(key=sort_key)

    print("Per-chapter counts:")
    report_counts(figures)

    manifest = {
        "source": "Bishop, Pattern Recognition and Machine Learning (2006)",
        "extractedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(figures),
        "figures": figures,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_PATH} ({len(figures)} figures)")


if __name__ == "__main__":
    sys.exit(main())
