import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const FIXTURES = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../tools/golden/fixtures');

/**
 * Loads a fixture emitted by a `tools/golden/gen_*.py` script.
 *
 * Every expected value in this package comes from here. A test that writes its own
 * expected numbers proves only that the implementation agrees with whoever typed them,
 * which for a Bessel ratio or an incomplete beta is worth nothing.
 */
export function loadFixture<T>(name: string): T {
  const path = resolve(FIXTURES, `${name}.json`);
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as T;
  } catch (cause) {
    throw new Error(
      `missing golden fixture "${name}". Generate it with \`node tools/golden/generate.mjs ${name}\`.`,
      { cause },
    );
  }
}

/** Relative where the expected value is large, absolute near zero, matching numpy's `allclose`. */
export function closeTo(actual: number, expected: number, tol = 1e-9): boolean {
  if (!Number.isFinite(expected)) return Object.is(actual, expected);
  return Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected));
}
