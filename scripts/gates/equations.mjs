#!/usr/bin/env node
/**
 * Gate 4: every equations[].id in every section's frontmatter must exist in
 * tools/extract/equations.json. Also reports, as information rather than
 * failure, manifest equations for a covered chapter that no section
 * references.
 */
import fs from 'node:fs';
import path from 'node:path';
import { parseGateArgs, reportGate } from './lib/cli.mjs';
import { findFiles } from './lib/walk.mjs';
import { readMdxFile } from './lib/mdx.mjs';
import { SECTIONS_DIR, EQUATIONS_MANIFEST } from './lib/paths.mjs';

export async function runEquationsGate({
  sectionsDir = SECTIONS_DIR,
  manifestPath = EQUATIONS_MANIFEST,
} = {}) {
  if (!fs.existsSync(manifestPath)) {
    throw new Error(
      `equation manifest not found at ${manifestPath}. Run ` +
        `\`node tools/extract/extract_equations.py\` first.`,
    );
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

  // Chapter 12 sits behind an OCR overlay in the source PDF, so ten of its equation
  // numbers cannot be detected and are supplied separately. Without merging them here the
  // gate would reject correct chapter 12 citations.
  const supplementPath = manifestPath.replace(/equations\.json$/, 'equations-supplement.json');
  if (fs.existsSync(supplementPath)) {
    const supplement = JSON.parse(fs.readFileSync(supplementPath, 'utf8'));
    manifest.equations = [...manifest.equations, ...supplement.equations];
  }

  const manifestIds = new Set(manifest.equations.map((e) => e.id));
  const idsByChapter = new Map();
  for (const eq of manifest.equations) {
    if (!idsByChapter.has(eq.chapter)) idsByChapter.set(eq.chapter, new Set());
    idsByChapter.get(eq.chapter).add(eq.id);
  }

  const files = findFiles(sectionsDir, '.mdx');
  const failures = [];
  const warnings = [];
  const info = [];
  const referencedIds = new Set();
  const chaptersCovered = new Set();

  if (files.length === 0) {
    info.push(`0 sections reference the manifest's ${manifest.equations.length} equations; nothing to check yet.`);
    return { gate: 'equations', ok: true, checked: 0, vacuous: true, failures, warnings, info };
  }

  for (const file of files) {
    const rel = path.relative(sectionsDir, file);
    const { hasFrontmatter, frontmatter, frontmatterError } = readMdxFile(file);

    if (!hasFrontmatter || frontmatterError) {
      failures.push({ file: rel, message: frontmatterError ?? 'no frontmatter block' });
      continue;
    }

    if (typeof frontmatter?.chapter === 'number') {
      chaptersCovered.add(frontmatter.chapter);
    }

    for (const eq of frontmatter?.equations ?? []) {
      if (!eq?.id) continue;
      referencedIds.add(eq.id);
      if (!manifestIds.has(eq.id)) {
        failures.push({ file: rel, message: `equation id "${eq.id}" is not in the equation manifest` });
      }
    }
  }

  for (const chapter of [...chaptersCovered].sort((a, b) => a - b)) {
    const chapterIds = idsByChapter.get(chapter);
    if (!chapterIds) continue;
    const unreferenced = [...chapterIds].filter((id) => !referencedIds.has(id));
    if (unreferenced.length > 0) {
      info.push(
        `chapter ${chapter}: ${unreferenced.length} manifest equation(s) not referenced by any section: ` +
          unreferenced.sort(compareEquationIds).join(', '),
      );
    }
  }

  return {
    gate: 'equations',
    ok: failures.length === 0,
    checked: files.length,
    failures,
    warnings,
    info,
  };
}

function compareEquationIds(a, b) {
  const [aMaj, aMin] = a.split('.').map(Number);
  const [bMaj, bMin] = b.split('.').map(Number);
  return aMaj - bMaj || aMin - bMin;
}

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  let result;
  try {
    result = await runEquationsGate();
  } catch (err) {
    console.error(`gate: equations\nERROR: ${err.message}`);
    process.exit(1);
  }
  process.exit(reportGate(result, { json: args.json }));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
