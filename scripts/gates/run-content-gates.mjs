#!/usr/bin/env node
import { parseGateArgs } from './lib/cli.mjs';
import { runFrontmatterGate } from './frontmatter.mjs';
import { runMdxStructureGate } from './mdx-structure.mjs';
import { runEquationsGate } from './equations.mjs';
import { runGraphGate } from './graph.mjs';

const GATES = [
  { name: 'frontmatter', run: runFrontmatterGate },
  { name: 'mdx-structure', run: runMdxStructureGate },
  { name: 'equations', run: runEquationsGate },
  { name: 'graph', run: runGraphGate },
];

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  const results = [];

  for (const { name, run } of GATES) {
    try {
      results.push(await run());
    } catch (err) {
      results.push({ gate: name, ok: false, checked: 0, failures: [{ message: err.message }], warnings: [], info: [] });
    }
  }

  if (args.json) {
    console.log(JSON.stringify(results, null, 2));
  } else {
    for (const result of results) {
      console.log(`\n=== gate: ${result.gate} ===`);
      if (result.vacuous) {
        console.log('0 files checked - nothing to validate yet (this still passes).');
      } else {
        console.log(`checked ${result.checked} file${result.checked === 1 ? '' : 's'}`);
      }
      for (const line of result.info ?? []) console.log(`  info: ${line}`);
      for (const w of result.warnings ?? []) console.log(`  warn: ${formatEntry(w)}`);
      for (const f of result.failures ?? []) console.log(`  FAIL: ${formatEntry(f)}`);
    }

    console.log('\n=== summary ===');
    const width = Math.max(...results.map((r) => r.gate.length));
    for (const result of results) {
      const status = result.ok ? 'PASS' : `FAIL (${result.failures.length})`;
      console.log(`  ${result.gate.padEnd(width)}  ${status}`);
    }
  }

  process.exit(results.every((r) => r.ok) ? 0 : 1);
}

function formatEntry(entry) {
  if (typeof entry === 'string') return entry;
  const { file, line, message } = entry;
  const loc = file ? (line ? `${file}:${line}` : file) : null;
  return loc ? `${loc} - ${message}` : message;
}

main();
