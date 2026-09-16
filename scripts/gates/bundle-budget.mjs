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

  // Matched by route rather than by <title>, because the rendered title is decorated with
  // the section number and the site name while the route is the section's identity. Title
  // matching was the earlier approach and broke the moment the layout added a suffix.
  const byRoute = new Map();
  for (const page of pages) {
    byRoute.set(routeOf(page, distDir), page);
  }

  const measured = [];
  for (const section of sections) {
    const page = byRoute.get(`sections/${section.id.replace('.', '-')}`);
    if (!page) {
      failures.push({
        file: section.file,
        message: `section ${section.id} has no built page at /sections/${section.id.replace('.', '-')}/ in dist/`,
      });
      continue;
    }
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

const ISLAND_URL_RE = /(?:component-url|renderer-url)="([^"]+)"/g;
const IMPORT_RE = /(?:^|[\s;}])(?:import|export)\s*(?:[\w*{},\s]*?from\s*)?["']([^"']+)["']|\bimport\(\s*["']([^"']+)["']\s*\)/g;

/**
 * Every byte of JavaScript the page causes the browser to fetch, gzipped.
 *
 * Island chunks are named in `astro-island` attributes rather than in a `<script src>`, and
 * each chunk statically imports further chunks, so counting script tags alone reports a
 * fraction of the truth: a page shipping the whole React runtime through an island measured
 * as a few kilobytes before this followed the graph.
 */
function collectJsBytes(entries, distDir) {
  const seen = new Set();
  const queue = [...entries];
  let bytes = 0;

  while (queue.length > 0) {
    const file = queue.pop();
    if (seen.has(file) || !fs.existsSync(file)) continue;
    seen.add(file);

    const source = fs.readFileSync(file);
    bytes += zlib.gzipSync(source).length;

    const text = source.toString('utf8');
    for (const match of text.matchAll(IMPORT_RE)) {
      const spec = match[1] ?? match[2];
      if (!spec || /^https?:\/\//.test(spec)) continue;
      const resolved = spec.startsWith('/')
        ? path.join(distDir, spec)
        : path.resolve(path.dirname(file), spec);
      if (resolved.endsWith('.js')) queue.push(resolved);
    }
  }

  return bytes;
}

function measurePage(htmlFile, distDir) {
  const html = fs.readFileSync(htmlFile, 'utf8');
  const titleMatch = TITLE_RE.exec(html);
  const title = titleMatch ? titleMatch[1].trim() : null;

  const entries = new Set();
  const add = (src) => {
    if (!src || /^https?:\/\//.test(src)) return;
    const resolved = src.startsWith('/')
      ? path.join(distDir, src)
      : path.join(path.dirname(htmlFile), src);
    if (fs.existsSync(resolved)) entries.add(resolved);
  };

  for (const match of html.matchAll(SCRIPT_SRC_RE)) add(match[1]);
  for (const match of html.matchAll(ISLAND_URL_RE)) add(match[1]);

  return { file: htmlFile, title, bytes: collectJsBytes([...entries], distDir) };
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

function routeOf(page, distDir) {
  return path
    .relative(distDir, page.file)
    .replace(/\\/g, '/')
    .replace(/\/?index\.html$/, '');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
