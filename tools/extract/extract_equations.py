"""
Build the committed equation-number manifest that the content-agent CI gate
checks frontmatter `equations: [{id: "3.49"}]` claims against.

Run: uv run --with pymupdf python tools/extract/extract_equations.py
Reads: the PDF named by $PRML_PDF (default: ../Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, one level above the repo)
Writes: tools/extract/equations.json (committed)
"""

import json
import re
import sys
from datetime import datetime, timezone

import fitz

from _pdf_toc import PDF_PATH, derive_offset, parse_toc_chapter_starts

OUT_PATH = __file__.rsplit("/", 1)[0] + "/equations.json"

CANDIDATE_RE = re.compile(r"\(\d{1,2}\.\d{1,3}\)")

# Body paragraphs start each line at a left margin around x=121 on a 504pt-wide
# page, and any inline citation "(3.49)" embedded in prose either has trailing
# words after it or sits on a line whose leading text starts at that same body
# margin. A genuine display-equation label's line, by contrast, starts well
# past the page midpoint: either the label is alone on the line, or the only
# thing sharing its line is a leftover fragment of the equation's own math
# notation (e.g. a trailing symbol from a fraction), which itself renders far
# right of the body margin. So a match only counts as a real label if nothing
# follows it on the line (no trailing prose, no closing period) and the line
# as a whole starts and ends in the right-hand margin band, not at the body's
# left margin. This still lets one rare case through: a citation that happens
# to be the last, trailing word of a short final paragraph line reaching well
# into that margin band. `dedupe_keeping_first_occurrence` below cleans up the
# only instance of that seen in this book (a duplicate "2.142").
LEFT_MIN_FRACTION = 0.55
RIGHT_MARGIN_FRACTION = 0.85

# A short formula can fit entirely on the same line as its trailing number,
# starting well left of LEFT_MIN_FRACTION (e.g. "tn(xn-u1)^T...+ const. (4.74)").
# That line is still its own dedicated block, isolated from surrounding prose,
# and short (a handful of lines at most) unlike a paragraph block or a long
# run of stacked list-form equations. Falling back to that structural signal
# recovers these without reopening the false-positive this margin rule exists
# to prevent, since a citation's line is never the sole/short block it lives in.
MAX_FALLBACK_BLOCK_LINES = 6


def find_equations(doc: fitz.Document, offset: int) -> tuple[list[dict], int]:
    equations = []
    rejected = 0

    for page_index in range(doc.page_count):
        page = doc[page_index]
        left_threshold = page.rect.width * LEFT_MIN_FRACTION
        right_threshold = page.rect.width * RIGHT_MARGIN_FRACTION
        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            last_line_index = len(block["lines"]) - 1
            is_short_block = last_line_index < MAX_FALLBACK_BLOCK_LINES
            for li, line in enumerate(block["lines"]):
                line_text = "".join(span["text"] for span in line["spans"]).strip()
                for match in CANDIDATE_RE.finditer(line_text):
                    is_trailing = match.end() == len(line_text)
                    is_right_margin = line["bbox"][2] >= right_threshold
                    meets_left_margin = line["bbox"][0] >= left_threshold
                    is_short_block_last_line = is_short_block and li == last_line_index
                    if (
                        is_trailing
                        and is_right_margin
                        and (meets_left_margin or is_short_block_last_line)
                    ):
                        eq_id = match.group()[1:-1]
                        chapter = int(eq_id.split(".")[0])
                        pdf_page = page_index + 1
                        equations.append(
                            {
                                "id": eq_id,
                                "chapter": chapter,
                                "pdfPage": pdf_page,
                                "bookPage": pdf_page - offset,
                            }
                        )
                    else:
                        rejected += 1

    return equations, rejected


def dedupe_keeping_first_occurrence(equations: list[dict]) -> list[dict]:
    """Bishop never reuses an equation number, so a duplicate id means one
    occurrence is a positional false positive (an inline citation that slipped
    past the margin heuristic). The genuine definition is always cited after
    it is introduced, so the earliest page in the pair is the real one."""
    seen: dict[str, dict] = {}
    dropped = []
    for eq in sorted(equations, key=lambda e: e["pdfPage"]):
        if eq["id"] in seen:
            dropped.append(eq)
            continue
        seen[eq["id"]] = eq
    if dropped:
        print(
            f"Dropped {len(dropped)} duplicate id(s) as false positives (kept earliest page):"
        )
        for eq in dropped:
            print(
                f"  {eq['id']} at pdfPage {eq['pdfPage']} (kept pdfPage {seen[eq['id']]['pdfPage']})"
            )
    return list(seen.values())


def sort_key(eq: dict) -> tuple[int, float]:
    sub = eq["id"].split(".", 1)[1]
    return (eq["chapter"], int(sub))


def check_contiguity(equations: list[dict]) -> None:
    by_chapter: dict[int, list[int]] = {}
    for eq in equations:
        sub = int(eq["id"].split(".", 1)[1])
        by_chapter.setdefault(eq["chapter"], []).append(sub)

    for chapter in sorted(by_chapter):
        subs = sorted(set(by_chapter[chapter]))
        raw = by_chapter[chapter]
        dup_counts = {n: raw.count(n) for n in set(raw) if raw.count(n) > 1}
        gaps = []
        expected = 1
        for s in subs:
            if s != expected:
                gaps.append((expected, s - 1))
            expected = s + 1
        status = []
        if subs[0] != 1:
            status.append(f"does not start at 1 (starts at {subs[0]})")
        if gaps:
            status.append(f"gaps: {gaps}")
        if dup_counts:
            status.append(f"duplicates: {dup_counts}")
        summary = "; ".join(status) if status else "contiguous"
        print(
            f"  chapter {chapter}: {len(subs)} unique ids, range {subs[0]}-{subs[-1]} ({summary})"
        )


def main() -> None:
    doc = fitz.open(PDF_PATH)
    chapter_starts = parse_toc_chapter_starts(doc)
    offset = derive_offset(doc, chapter_starts)
    print(f"Using pdf_page = book_page + {offset}")

    equations, rejected = find_equations(doc, offset)
    print(
        f"Found {len(equations)} equation labels, rejected {rejected} positional candidates "
        f"(inline citations mid-paragraph)"
    )

    equations = dedupe_keeping_first_occurrence(equations)
    equations.sort(key=sort_key)

    print("Contiguity check per chapter:")
    check_contiguity(equations)

    manifest = {
        "source": "Bishop, Pattern Recognition and Machine Learning (2006)",
        "extractedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(equations),
        "equations": equations,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT_PATH} ({len(equations)} equations)")


if __name__ == "__main__":
    sys.exit(main())
