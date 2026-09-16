#!/usr/bin/env node
/**
 * Gate 8: per-page JavaScript ceiling over apps/web/dist/.
 *
 * Prose pages (frontmatter.widgets is empty) must ship 0 bytes of page JS.
 * Widget pages get a gzip-bytes budget from budgets.json (defaultWidgetBudgetBytes,
 * overridable per section id in `overrides`). See README for how the default
 * was chosen. Pages are matched to sections by <title> for the same reason as
 * gate 7: route URL shape is not this gate's to decide.
 */
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { parseGateArgs, reportGate } from './lib/cli.mjs';
import { findFiles } from './lib/walk.mjs';
import { readMdxFile } from './lib/mdx.mjs';
import { SECTIONS_DIR, DIST_DIR, BUDGETS_JSON } from './lib/paths.mjs';

const TITLE_RE = /<title[^>]*>([\s\S]*?)<\/title>/i;
const SCRIPT_SRC_RE = /<script\b[^>]*\bsrc=["']([^"']+)["'][^>]*>/gi;

export async function runBundleBudgetGate({
  sectionsDir = SECTIONS_DIR,
  distDir = DIST_DIR,
  budgetsPath = BUDGETS_JSON,
} = {}) {
  if (!fs.existsSync(distDir)) {
    throw new Error(`${path.relative(process.cwd(), distDir)} not found. Run \`pnpm build\` first.`);
  }

  const budgets = JSON.parse(fs.readFileSync(budgetsPath, 'utf8'));

  const htmlFiles = findFiles(distDir, '.html').filter(
    (f) => !f.includes(`${path.sep}pagefind${path.sep}`),
  );
  const pages = htmlFiles.map((file) => measurePage(file, distDir));

  const files = findFiles(sectionsDir, '.mdx');
  const failures = [];
  const warnings = [];
  const info = [];

  const sections = [];
  for (const file of files) {
    const rel = path.relative(sectionsDir, file);
    const { hasFrontmatter, frontmatter, frontmatterError } = readMdxFile(file);
    if (!hasFrontmatter || frontmatterError || !frontmatter?.title) {
      failures.push({ file: rel, message: frontmatterError ?? 'no usable frontmatter (missing "title")' });
      continue;
    }
    if (frontmatter.draft) continue;
    sections.push({
      file: rel,
      id: frontmatter.section,
      title: frontmatter.title,
      isWidgetPage: (frontmatter.widgets ?? []).length > 0,
    });
  }

  if (sections.length === 0) {
    return {
      gate: 'bundle-budget',
      ok: failures.length === 0,
      checked: 0,
      vacuous: true,
      failures,
      warnings,
      info,
    };
  }

  const byTitle = new Map();
  for (const page of pages) {
    if (!page.title) continue;
    if (!byTitle.has(page.title)) byTitle.set(page.title, []);
    byTitle.get(page.title).push(page);
  }

  const measured = [];
  for (const section of sections) {
    const matches = byTitle.get(section.title) ?? [];
    if (matches.length === 0) {
      failures.push({ file: section.file, message: `"${section.title}" has no built HTML page in dist/` });
      continue;
    }
    if (matches.length > 1) {
      warnings.push({
        file: section.file,
        message: `title "${section.title}" matches ${matches.length} built pages; skipping strict check`,
      });
      continue;
    }
    const [page] = matches;
    const budget = section.isWidgetPage
      ? (budgets.overrides?.[section.id] ?? budgets.defaultWidgetBudgetBytes)
      : budgets.proseBudgetBytes;

    measured.push({ ...section, bytes: page.bytes, budget });

    if (page.bytes > budget) {
      failures.push({
        file: section.file,
        message:
          `${section.isWidgetPage ? 'widget' : 'prose'} page ships ${page.bytes} gzip bytes of JS, ` +
          `budget is ${budget}`,
      });
    }
  }

  for (const m of measured.sort((a, b) => b.bytes - a.bytes)) {
    info.push(`${m.id ?? m.file}: ${m.bytes}B / ${m.budget}B budget (${m.isWidgetPage ? 'widget' : 'prose'})`);
  }

  return {
    gate: 'bundle-budget',
    ok: failures.length === 0,
    checked: sections.length,
    failures,
    warnings,
    info,
  };
}

function measurePage(htmlFile, distDir) {
  const html = fs.readFileSync(htmlFile, 'utf8');
  const titleMatch = TITLE_RE.exec(html);
  const title = titleMatch ? titleMatch[1].trim() : null;

  const scriptFiles = new Set();
  for (const match of html.matchAll(SCRIPT_SRC_RE)) {
    const src = match[1];
    if (/^https?:\/\//.test(src)) continue;
    const resolved = src.startsWith('/')
      ? path.join(distDir, src)
      : path.join(path.dirname(htmlFile), src);
    if (fs.existsSync(resolved)) scriptFiles.add(resolved);
  }

  let bytes = 0;
  for (const scriptFile of scriptFiles) {
    bytes += zlib.gzipSync(fs.readFileSync(scriptFile)).length;
  }

  return { file: htmlFile, title, bytes };
}

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  let result;
  try {
    result = await runBundleBudgetGate();
  } catch (err) {
    console.error(`gate: bundle-budget\nERROR: ${err.message}`);
    process.exit(1);
  }
  process.exit(reportGate(result, { json: args.json }));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
