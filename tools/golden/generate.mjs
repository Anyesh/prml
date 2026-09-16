#!/usr/bin/env node
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));

function discoverNames() {
  return readdirSync(HERE)
    .filter((f) => f.startsWith('gen_') && f.endsWith('.py'))
    .map((f) => f.slice('gen_'.length, -'.py'.length))
    .sort();
}

function runOne(name) {
  const script = resolve(HERE, `gen_${name}.py`);
  console.log(`[golden] generating ${name}...`);
  const result = spawnSync('uv', ['run', '--with', 'numpy', '--with', 'scipy', 'python', script], {
    stdio: 'inherit',
    cwd: HERE,
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`generating fixture "${name}" failed (exit code ${result.status})`);
  }
}

const requested = process.argv.slice(2);
const names = requested.length > 0 ? requested : discoverNames();

if (names.length === 0) {
  console.error('[golden] no gen_*.py scripts found');
  process.exit(1);
}

for (const name of names) {
  runOne(name);
}

console.log(`[golden] done: ${names.join(', ')}`);
