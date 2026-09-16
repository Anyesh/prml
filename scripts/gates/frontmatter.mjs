#!/usr/bin/env node
/**
 * Gate 1: validates every content/sections/**\/*.mdx frontmatter against the
 * frozen zod schema in apps/web/src/content.config.ts (imported, not
 * duplicated - see lib/schema.mjs), plus two checks zod cannot express:
 *   - the file's path encodes the same chapter/section as its frontmatter
 *   - no two files claim the same section number
 */
import path from 'node:path';
import { parseGateArgs, reportGate } from './lib/cli.mjs';
import { findFiles } from './lib/walk.mjs';
import { readMdxFile } from './lib/mdx.mjs';
import { loadCollections } from './lib/schema.mjs';
import { SECTIONS_DIR } from './lib/paths.mjs';

const DIR_RE = /^(\d{2})-[a-z0-9]+(?:-[a-z0-9]+)*$/;
const FILE_RE = /^(\d{1,2}\.\d{1,2})-[a-z0-9]+(?:-[a-z0-9]+)*\.mdx$/;

export async function runFrontmatterGate({ sectionsDir = SECTIONS_DIR } = {}) {
  const files = findFiles(sectionsDir, '.mdx');
  const failures = [];
  const warnings = [];
  const sectionOwners = new Map();

  if (files.length === 0) {
    return { gate: 'frontmatter', ok: true, checked: 0, vacuous: true, failures, warnings, info: [] };
  }

  const { sections } = await loadCollections();

  for (const file of files) {
    const rel = path.relative(sectionsDir, file);
    const { hasFrontmatter, frontmatter, frontmatterError, } = readMdxFile(file);

    if (!hasFrontmatter) {
      failures.push({ file: rel, message: 'no frontmatter block (missing leading --- ... ---)' });
      continue;
    }
    if (frontmatterError) {
      failures.push({ file: rel, message: `frontmatter is not valid YAML: ${frontmatterError}` });
      continue;
    }

    const result = sections.schema.safeParse(frontmatter);
    if (!result.success) {
      for (const issue of result.error.issues) {
        failures.push({ file: rel, message: `${issue.path.join('.') || '(root)'}: ${issue.message}` });
      }
      continue;
    }

    const data = result.data;

    const dirName = path.basename(path.dirname(file));
    const fileName = path.basename(file);
    const dirMatch = DIR_RE.exec(dirName);
    const fileMatch = FILE_RE.exec(fileName);

    if (!dirMatch) {
      failures.push({
        file: rel,
        message: `parent directory "${dirName}" must look like "03-linear-models-for-regression"`,
      });
    } else {
      const chapterFromPath = Number.parseInt(dirMatch[1], 10);
      if (chapterFromPath !== data.chapter) {
        failures.push({
          file: rel,
          message: `directory encodes chapter ${chapterFromPath} but frontmatter says chapter: ${data.chapter}`,
        });
      }
    }

    if (!fileMatch) {
      failures.push({
        file: rel,
        message: `filename "${fileName}" must look like "3.3-bayesian-linear-regression.mdx"`,
      });
    } else {
      const sectionFromPath = fileMatch[1];
      if (sectionFromPath !== data.section) {
        failures.push({
          file: rel,
          message: `filename encodes section "${sectionFromPath}" but frontmatter says section: "${data.section}"`,
        });
      }
    }

    if (sectionOwners.has(data.section)) {
      const other = sectionOwners.get(data.section);
      failures.push({
        file: rel,
        message: `section "${data.section}" is also claimed by ${other}`,
      });
    } else {
      sectionOwners.set(data.section, rel);
    }
  }

  return {
    gate: 'frontmatter',
    ok: failures.length === 0,
    checked: files.length,
    failures,
    warnings,
    info: [],
  };
}

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  let result;
  try {
    result = await runFrontmatterGate();
  } catch (err) {
    console.error(`gate: frontmatter\nERROR: ${err.message}`);
    process.exit(1);
  }
  process.exit(reportGate(result, { json: args.json }));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
