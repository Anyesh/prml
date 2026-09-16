import fs from 'node:fs';
import { parse as parseYaml } from 'yaml';

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/;

/**
 * Splits an MDX file into raw frontmatter text and body, tracking the 1-based
 * line number the body starts on so downstream regex offsets can be reported
 * as real file line numbers.
 */
export function readMdxFile(filePath) {
  const source = fs.readFileSync(filePath, 'utf8');
  const match = FRONTMATTER_RE.exec(source);
  if (!match) {
    return { hasFrontmatter: false, frontmatter: null, frontmatterError: null, body: source, bodyStartLine: 1 };
  }
  const frontmatterText = match[1];
  const body = source.slice(match[0].length);
  const bodyStartLine = countLines(match[0]) + 1;

  let frontmatter = null;
  let frontmatterError = null;
  try {
    frontmatter = parseYaml(frontmatterText);
  } catch (err) {
    frontmatterError = err.message;
  }

  return { hasFrontmatter: true, frontmatter, frontmatterError, body, bodyStartLine };
}

function countLines(text) {
  let count = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '\n') count++;
  }
  return count;
}

/** 1-based line number of `index` within `text`. */
export function lineAt(text, index) {
  return countLines(text.slice(0, index)) + 1;
}

/**
 * Blanks out fenced code blocks (```...```) while preserving line numbers,
 * so headings/math/component scans do not trip over example code.
 */
export function stripCodeFences(body) {
  return body.replace(/^ {0,3}```.*$[\s\S]*?^ {0,3}```\s*$/gm, (block) =>
    block.replace(/[^\n]/g, ' '),
  );
}
