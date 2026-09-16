#!/usr/bin/env node
/**
 * Gate 2: every section body must contain exactly these six H2 blocks, in
 * order, each exactly once (see README for the precise marker definition):
 *   Intuition, The math, Interactive, Worked example, Recall, Connections
 *
 * Plus:
 *   - Intuition holds no $-delimited math and no <MathBlock>
 *   - Interactive holds at least one widget element and a caption
 *   - Recall holds between 3 and 6 <Recall> elements
 *   - every <MathBlock id="x.y"> in the body is in frontmatter.equations, and
 *     vice versa
 */
import path from 'node:path';
import { parseGateArgs, reportGate } from './lib/cli.mjs';
import { findFiles } from './lib/walk.mjs';
import { readMdxFile, stripCodeFences, lineAt } from './lib/mdx.mjs';
import { SECTIONS_DIR } from './lib/paths.mjs';

export const REQUIRED_HEADINGS = [
  'intuition',
  'the math',
  'interactive',
  'worked example',
  'recall',
  'connections',
];

const HEADING_RE = /^##\s+(.+?)\s*$/gm;
const INLINE_MATH_RE = /\$[^$\n]+\$/;
const BLOCK_MATH_RE = /\$\$[\s\S]+?\$\$/;
const MATHBLOCK_TAG_RE = /<MathBlock\b/;
const MATHBLOCK_ID_RE = /<MathBlock\b[^>]*\bid=["']([^"']+)["']/g;
const RECALL_TAG_RE = /<Recall\b/g;
const JSX_COMPONENT_RE = /<([A-Z][A-Za-z0-9]*)\b/g;
// A widget's caption is authored in the section as a `caption=` prop rather than a
// separate element, because the frame renders exactly one caption and two would drift.
const CAPTION_RE = /<Caption\b|<figcaption\b|\bcaption=/i;

export async function runMdxStructureGate({ sectionsDir = SECTIONS_DIR } = {}) {
  const files = findFiles(sectionsDir, '.mdx');
  const failures = [];
  const warnings = [];

  if (files.length === 0) {
    return { gate: 'mdx-structure', ok: true, checked: 0, vacuous: true, failures, warnings, info: [] };
  }

  for (const file of files) {
    const rel = path.relative(sectionsDir, file);
    const { hasFrontmatter, frontmatter, frontmatterError, body, bodyStartLine } = readMdxFile(file);
    const scannable = stripCodeFences(body);

    const headings = [];
    for (const match of scannable.matchAll(HEADING_RE)) {
      headings.push({
        text: match[1].trim(),
        normalized: match[1].trim().toLowerCase(),
        index: match.index,
        line: bodyStartLine + lineAt(scannable, match.index) - 1,
      });
    }

    checkHeadingOrder(headings, rel, failures);

    const sections = sliceBySections(scannable, headings);

    const intuition = sections.get('intuition');
    if (intuition) {
      checkIntuitionHasNoMath(intuition, rel, bodyStartLine, failures);
    }

    const interactive = sections.get('interactive');
    if (interactive) {
      checkInteractiveHasWidgetAndCaption(interactive, rel, failures);
    }

    const recall = sections.get('recall');
    if (recall) {
      checkRecallCount(recall, rel, failures);
    }

    const equationsInBody = new Set();
    for (const match of scannable.matchAll(MATHBLOCK_ID_RE)) {
      equationsInBody.add(match[1]);
    }

    if (!hasFrontmatter) {
      failures.push({ file: rel, message: 'no frontmatter block; cannot cross-check MathBlock ids' });
    } else if (frontmatterError) {
      failures.push({ file: rel, message: `frontmatter is not valid YAML: ${frontmatterError}` });
    } else {
      checkMathBlockIdsMatchFrontmatter(equationsInBody, frontmatter, rel, failures);
    }
  }

  return {
    gate: 'mdx-structure',
    ok: failures.length === 0,
    checked: files.length,
    failures,
    warnings,
    info: [],
  };
}

function checkHeadingOrder(headings, rel, failures) {
  const seen = new Map();
  for (const heading of headings) {
    if (!REQUIRED_HEADINGS.includes(heading.normalized)) continue;
    if (seen.has(heading.normalized)) {
      failures.push({
        file: rel,
        line: heading.line,
        message: `duplicate "## ${heading.text}" heading (first seen at line ${seen.get(heading.normalized)})`,
      });
    } else {
      seen.set(heading.normalized, heading.line);
    }
  }

  const orderedFound = headings
    .map((h) => h.normalized)
    .filter((n) => REQUIRED_HEADINGS.includes(n));

  const missing = REQUIRED_HEADINGS.filter((h) => !seen.has(h));
  for (const name of missing) {
    failures.push({ file: rel, message: `missing required "## ${titleCase(name)}" heading` });
  }

  if (missing.length > 0) return;

  const expectedOrder = REQUIRED_HEADINGS;
  const actualFirstOccurrence = expectedOrder.map((name) =>
    orderedFound.indexOf(name),
  );
  for (let i = 1; i < actualFirstOccurrence.length; i++) {
    if (actualFirstOccurrence[i] < actualFirstOccurrence[i - 1]) {
      failures.push({
        file: rel,
        line: seen.get(expectedOrder[i]),
        message:
          `"## ${titleCase(expectedOrder[i])}" appears out of order ` +
          `(expected after "## ${titleCase(expectedOrder[i - 1])}")`,
      });
    }
  }
}

function titleCase(normalized) {
  return normalized.replace(/\b\w/g, (c) => c.toUpperCase());
}

function sliceBySections(body, headings) {
  const map = new Map();
  const required = headings.filter((h) => REQUIRED_HEADINGS.includes(h.normalized));
  for (let i = 0; i < required.length; i++) {
    const start = required[i].index;
    const end = i + 1 < required.length ? required[i + 1].index : body.length;
    if (!map.has(required[i].normalized)) {
      map.set(required[i].normalized, body.slice(start, end));
    }
  }
  return map;
}

function checkIntuitionHasNoMath(sectionText, rel, bodyStartLine, failures) {
  const inline = INLINE_MATH_RE.exec(sectionText);
  if (inline) {
    failures.push({
      file: rel,
      line: bodyStartLine + lineAt(sectionText, inline.index) - 1,
      message: 'Intuition section must contain no math, but has $-delimited math',
    });
  }
  const block = BLOCK_MATH_RE.exec(sectionText);
  if (block) {
    failures.push({
      file: rel,
      line: bodyStartLine + lineAt(sectionText, block.index) - 1,
      message: 'Intuition section must contain no math, but has $$-delimited math',
    });
  }
  const mathBlockTag = MATHBLOCK_TAG_RE.exec(sectionText);
  if (mathBlockTag) {
    failures.push({
      file: rel,
      line: bodyStartLine + lineAt(sectionText, mathBlockTag.index) - 1,
      message: 'Intuition section must contain no math, but has a <MathBlock>',
    });
  }
}

function checkInteractiveHasWidgetAndCaption(sectionText, rel, failures) {
  const componentTags = [...sectionText.matchAll(JSX_COMPONENT_RE)].map((m) => m[1]);
  const hasWidget = componentTags.some((name) => name !== 'Caption' && name !== 'MathBlock');
  if (!hasWidget) {
    failures.push({ file: rel, message: 'Interactive section has no widget element' });
  }
  if (!CAPTION_RE.test(sectionText)) {
    failures.push({ file: rel, message: 'Interactive section has no caption (<Caption> or <figcaption>)' });
  }
}

function checkRecallCount(sectionText, rel, failures) {
  const count = [...sectionText.matchAll(RECALL_TAG_RE)].length;
  if (count < 3 || count > 6) {
    failures.push({
      file: rel,
      message: `Recall section has ${count} <Recall> elements, expected 3-6`,
    });
  }
}

function checkMathBlockIdsMatchFrontmatter(idsInBody, frontmatter, rel, failures) {
  const idsInFrontmatter = new Set((frontmatter?.equations ?? []).map((e) => e?.id).filter(Boolean));

  for (const id of idsInBody) {
    if (!idsInFrontmatter.has(id)) {
      failures.push({
        file: rel,
        message: `<MathBlock id="${id}"> is not listed in frontmatter.equations`,
      });
    }
  }
  for (const id of idsInFrontmatter) {
    if (!idsInBody.has(id)) {
      failures.push({
        file: rel,
        message: `frontmatter.equations lists "${id}" but no <MathBlock id="${id}"> appears in the body`,
      });
    }
  }
}

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  const result = await runMdxStructureGate();
  process.exit(reportGate(result, { json: args.json }));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
