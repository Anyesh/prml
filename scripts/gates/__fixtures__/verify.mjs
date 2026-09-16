#!/usr/bin/env node
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runFrontmatterGate } from '../frontmatter.mjs';
import { runMdxStructureGate } from '../mdx-structure.mjs';
import { runEquationsGate } from '../equations.mjs';
import { runGraphGate } from '../graph.mjs';

const HERE = fileURLToPath(new URL('.', import.meta.url));

function allMessages(result) {
  return [...result.failures, ...result.info].map((entry) =>
    typeof entry === 'string' ? entry : entry.message,
  );
}

function expectCaught(name, result, expectedSubstrings) {
  const messages = allMessages(result);
  let ok = !result.ok;
  const missing = [];
  for (const expected of expectedSubstrings) {
    if (!messages.some((m) => m.includes(expected))) {
      ok = false;
      missing.push(expected);
    }
  }
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}`);
  if (!ok) {
    if (missing.length > 0) console.log(`  did not catch: ${JSON.stringify(missing)}`);
    if (result.ok) console.log('  gate reported ok:true, expected a failure');
    console.log('  actual failures:', JSON.stringify(result.failures, null, 2));
  }
  return ok;
}

async function main() {
  const results = [];

  const gate1 = await runFrontmatterGate({ sectionsDir: path.join(HERE, 'gate1-bad', 'sections') });
  results.push(
    expectCaught('gate1: frontmatter', gate1, [
      'difficulty',
      'also claimed by',
      'encodes section',
      'encodes chapter',
      'not valid YAML',
      'no frontmatter block',
    ]),
  );

  const gate2 = await runMdxStructureGate({ sectionsDir: path.join(HERE, 'gate2-bad', 'sections') });
  results.push(
    expectCaught('gate2: mdx-structure', gate2, [
      'missing required "## Recall"',
      'appears out of order',
      'duplicate "## Intuition"',
      'must contain no math',
      'no widget element',
      'no caption',
      'expected 3-6',
      'is not listed in frontmatter.equations',
      'no <MathBlock',
    ]),
  );

  const gate4 = await runEquationsGate({
    sectionsDir: path.join(HERE, 'gate4-bad', 'sections'),
    manifestPath: path.join(HERE, 'gate4-bad', 'equations.json'),
  });
  results.push(
    expectCaught('gate4: equations', gate4, [
      '"1.999" is not in the equation manifest',
      '1.2',
    ]),
  );

  const gate5 = await runGraphGate({
    sectionsDir: path.join(HERE, 'gate5-bad', 'sections'),
    conceptsDir: path.join(HERE, 'gate5-bad', 'concepts'),
    rootsPath: path.join(HERE, 'gate5-bad', 'roots.json'),
  });
  results.push(
    expectCaught('gate5: graph', gate5, [
      'prereq cycle',
      'does not exist',
      'unknown concept "nonexistent-concept"',
      'requires-cycle',
      'introducedIn "7.7" does not exist',
      'orphaned',
      'is already defined in concepts/ch03.yaml',
    ]),
  );

  const allPass = results.every(Boolean);
  console.log(allPass ? '\nall fixtures caught as expected' : '\nsome fixtures were NOT caught');
  process.exit(allPass ? 0 : 1);
}

main();
