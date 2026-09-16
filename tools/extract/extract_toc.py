"""
Dumps every top-level `x.y` section in the book with its title and printed page range.

The section is the content unit of the site (one MDX file per `x.y`), so chapter authors
need the exact list and its page ranges before writing anything. `x.y.z` entries are
recorded as children because they live inside their parent's page rather than getting one.

Usage:
    uv run --with pymupdf python tools/extract/extract_toc.py
"""

import json
import re
from pathlib import Path

import fitz

from _pdf_toc import (
    PDF_PATH,
    derive_offset,
    parse_toc_appendix_a_start,
    parse_toc_chapter_starts,
)

OUT_PATH = Path(__file__).resolve().parent / "out" / "toc-sections.json"

# The embedded outline titles carry their own numbering, in two shapes: "3.1. Linear Basis
# Function Models" at level 2 and "3.1.1 Some useful basis functions" at level 3.
TITLE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})(?:\.(\d{1,2}))?\.?\s+(.*)$")


def parse_outline(doc: fitz.Document, offset: int) -> list[dict]:
    entries = []
    for level, title, pdf_page in doc.get_toc():
        if level not in (2, 3):
            continue
        match = TITLE_RE.match(title.replace("\r", " ").strip())
        if not match:
            continue
        chapter, minor, sub, text = match.groups()
        entries.append(
            {
                "id": f"{chapter}.{minor}" + (f".{sub}" if sub else ""),
                "chapter": int(chapter),
                "minor": int(minor),
                "depth": 3 if sub else 2,
                "title": text.strip(),
                "startPage": pdf_page - offset,
            }
        )
    return entries


def group_by_section(entries: list[dict], chapter_ends: dict[int, int]) -> list[dict]:
    tops = {e["id"]: {**e, "children": []} for e in entries if e["depth"] == 2}

    for entry in entries:
        if entry["depth"] != 3:
            continue
        parent = ".".join(entry["id"].split(".")[:2])
        if parent in tops:
            tops[parent]["children"].append(
                {"id": entry["id"], "title": entry["title"]}
            )

    ordered = sorted(tops.values(), key=lambda e: (e["chapter"], e["minor"]))
    for index, entry in enumerate(ordered):
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        if following and following["chapter"] == entry["chapter"]:
            entry["bookPages"] = [entry["startPage"], following["startPage"] - 1]
        else:
            entry["bookPages"] = [entry["startPage"], chapter_ends[entry["chapter"]]]
        for key in ("startPage", "depth", "minor"):
            del entry[key]
    return ordered


def main() -> None:
    doc = fitz.open(PDF_PATH)
    chapter_starts = parse_toc_chapter_starts(doc)
    appendix_start = parse_toc_appendix_a_start(doc)
    offset = derive_offset(doc, chapter_starts)

    chapter_ends = {
        number: chapter_starts.get(number + 1, appendix_start) - 1
        for number in sorted(chapter_starts)
    }

    sections = group_by_section(parse_outline(doc, offset), chapter_ends)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "count": len(sections),
                "chapterStarts": chapter_starts,
                "chapterEnds": chapter_ends,
                "sections": sections,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{len(sections)} top-level sections -> {OUT_PATH}")


if __name__ == "__main__":
    main()
