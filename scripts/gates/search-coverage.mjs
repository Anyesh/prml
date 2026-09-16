#!/usr/bin/env node
/**
 * Gate 7: after `pnpm build`, every non-draft section must appear in the
 * Pagefind index with a non-empty body.
 *
 * Pagefind's fragment files are gzip-compressed JSON prefixed with a
 * "pagefind_dcd" magic header (verified empirically against a real pagefind
 * build; see README). Route URLs aren't decided by this gate's owner, so
 * sections are matched to fragments by title rather than by URL shape.
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { parseGateArgs, reportGate } from './lib/cli.mjs';
import { findFiles } from './lib/walk.mjs';
import { readMdxFile } from './lib/mdx.mjs';
import { SECTIONS_DIR, DIST_DIR } from './lib/paths.mjs';

export async function runSearchCoverageGate({ sectionsDir = SECTIONS_DIR, distDir = DIST_DIR } = {}) {
  if (!fs.existsSync(distDir)) {
    throw new Error(`${path.relative(process.cwd(), distDir)} not found. Run \`pnpm build\` first.`);
  }

  const pagefindDir = path.join(distDir, 'pagefind');
  const entryPath = path.join(pagefindDir, 'pagefind-entry.json');
  if (!fs.existsSync(entryPath)) {
    throw new Error(
      `${path.relative(process.cwd(), entryPath)} not found. Run \`pnpm build\` first ` +
        `(the build's "index" step runs the Pagefind indexer).`,
    );
  }

  const entry = JSON.parse(fs.readFileSync(entryPath, 'utf8'));
  const fragments = readFragments(pagefindDir);

  const files = findFiles(sectionsDir, '.mdx');
  const failures = [];
  const warnings = [];
  const info = [`pagefind ${entry.version}, ${fragments.length} fragment(s) indexed`];

  const sections = [];
  for (const file of files) {
    const rel = path.relative(sectionsDir, file);
    const { hasFrontmatter, frontmatter, frontmatterError } = readMdxFile(file);
    if (!hasFrontmatter || frontmatterError || !frontmatter?.title) {
      failures.push({ file: rel, message: frontmatterError ?? 'no usable frontmatter (missing "title")' });
      continue;
    }
    if (frontmatter.draft) continue;
    sections.push({ file: rel, title: frontmatter.title });
  }

  if (sections.length === 0) {
    return {
      gate: 'search-coverage',
      ok: failures.length === 0,
      checked: 0,
      vacuous: true,
      failures,
      warnings,
      info,
    };
  }

  const byTitle = new Map();
  for (const fragment of fragments) {
    const title = fragment.meta?.title;
    if (!title) continue;
    if (!byTitle.has(title)) byTitle.set(title, []);
    byTitle.get(title).push(fragment);
  }

  for (const section of sections) {
    const matches = byTitle.get(section.title) ?? [];
    if (matches.length === 0) {
      failures.push({ file: section.file, message: `"${section.title}" was not found in the Pagefind index` });
      continue;
    }
    if (matches.length > 1) {
      warnings.push({
        file: section.file,
        message: `title "${section.title}" matches ${matches.length} indexed pages; skipping strict check`,
      });
      continue;
    }
    const [fragment] = matches;
    if (!fragment.content || fragment.content.trim().length === 0) {
      failures.push({ file: section.file, message: `"${section.title}" is indexed with an empty body` });
    }
  }

  return {
    gate: 'search-coverage',
    ok: failures.length === 0,
    checked: sections.length,
    failures,
    warnings,
    info,
  };
}

function readFragments(pagefindDir) {
  const fragmentDir = path.join(pagefindDir, 'fragment');
  const fragmentFiles = findFiles(fragmentDir, '.pf_fragment');
  const fragments = [];
  for (const file of fragmentFiles) {
    const decompressed = zlib.gunzipSync(fs.readFileSync(file)).toString('utf8');
    const jsonStart = decompressed.indexOf('{');
    fragments.push(JSON.parse(decompressed.slice(jsonStart)));
  }
  return fragments;
}

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  let result;
  try {
    result = await runSearchCoverageGate();
  } catch (err) {
    console.error(`gate: search-coverage\nERROR: ${err.message}`);
    process.exit(1);
  }
  process.exit(reportGate(result, { json: args.json }));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
