# tools/extract

Scripts that pull structured data out of the source PDF
(the PDF named by $PRML_PDF (default: ../Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf, one level above the repo),
outside this repo). PRML is free for personal use but not redistributable, so
book prose never gets committed; only numbers (page indices, equation ids,
figure ids) are, since those aren't the author's expression.

All three scripts run through `uv`:

```
uv run --with pymupdf python tools/extract/extract_text.py
uv run --with pymupdf python tools/extract/extract_equations.py
uv run --with pymupdf python tools/extract/extract_figures.py
```

`_pdf_toc.py` is a shared internal module (not run directly) that parses the
table of contents (PDF pages 8-24) once and derives the physical-page/book-page
offset, so all three scripts agree on page numbers.

## extract_text.py

When: run once, or whenever a chapter's raw text is needed for authoring a
content page and citing book pages.

Reads: the PDF's text layer via PyMuPDF (not `pdftotext`, because step 2 needs
per-span coordinates that poppler discards).

Writes: `tools/extract/out/front.txt`, `ch01.txt`..`ch14.txt`, `appendix.txt`
(16 files), each page prefixed with a `===== page NNN =====` marker (NNN = the
PDF's 1-based physical page index).

Committed: no. `out/` is gitignored in full, since this is the copyrighted
book text and must never enter the repo.

## extract_equations.py

When: run whenever the source PDF changes, or the detection logic changes.
Content agents' frontmatter (`equations: [{id: "3.49"}]`) is checked against
this manifest by a CI gate, so it must be regenerated (and re-verified) before
that gate can trust new ids.

Reads: the PDF, plus `_pdf_toc.py`.

Writes: `tools/extract/equations.json`.

Committed: yes. It holds only equation numbers and page indices, not equation
content.

Detection: a display equation's number is flush right in its own text block,
isolated from body paragraphs (which run to a left margin around x=121 on a
504pt-wide page). The script requires the match to be the last thing on its
line, with no trailing prose or period, and requires the line to reach both a
right-margin band and, with a short-block fallback for numbers that share a
line with the tail of their own formula, a left-margin band too. That pairing
is what separates a genuine label from an inline citation like "(3.49)"
appearing mid-paragraph. Duplicate ids get deduped by keeping the earliest
page, since Bishop never reuses a number, so a duplicate always pairs one
genuine definition with a false-positive citation.

Known gap: 10 of roughly 1958 equations in Chapter 12 (12.22, 12.34, 12.35,
12.53, 12.65 through 12.68, 12.74, 12.75) are unrecoverable from this PDF's
text layer. Pages 581 through 620 embed a `HiddenHorzOCR` font instead of the
vector Times-Roman text used everywhere else, so that page range is a
scanned-and-OCR'd overlay rather than the book's real text layer, and OCR
misreads characters there; "PCA" reads back as "peA" and a "5" reads back as
"~" on those pages. This is a source-PDF defect confined to that exact page
range rather than a detector bug, confirmed by scanning every page in the
document for that font name and finding it nowhere else.

## extract_figures.py

When: same cadence as extract_equations.py.

Reads: the PDF, plus `_pdf_toc.py`.

Writes: `tools/extract/figures.json`. Caption text is never stored, only the
figure number and page, because captions are Bishop's prose and this project
writes its own descriptions.

Committed: yes, for the same reason as equations.json.

Detection: a caption's own first line starts with `Figure x.y` and sits at a
narrow outer margin around 30 to 70 points, distinct from the roughly 121pt
body margin where an inline mention such as "Figure 3.11." as the first word
of a sentence would start.

Confidence: 282 figures found, well short of the roughly 430 figures a full
PRML print run is commonly said to have. This was cross-checked against a hard
ceiling: every literal occurrence of "Figure x.y" anywhere in this PDF's
extracted text, captions and inline mentions combined, comes to only 291
unique figure ids in this copy's text layer, and 282 of those 291 were
captured as genuine captions. The remaining 9 are all in Chapter 12 and trace
to the same OCR-scanned page range described above. If this specific PDF is
missing figures that the roughly 430 estimate assumes, they are not
recoverable from its text layer by any caption-detection approach; the gap
sits in the source file, not in this script.
