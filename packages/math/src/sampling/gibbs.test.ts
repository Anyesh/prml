import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { gibbsSampleMvn, overRelaxationStep } from './gibbs.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_gibbs');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('overRelaxationStep', () => {
  it('matches the closed form (PRML 11.50) exactly', () => {
    for (const c of casesFor('overRelaxationStep')) {
      const actual = overRelaxationStep(c['zi'] as number, c['mean'] as number, c['variance'] as number, c['alpha'] as number, c['nu'] as number);
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
  });

  it('reduces to a plain Gibbs draw at alpha = 0', () => {
    expect(overRelaxationStep(5, 1, 4, 0, 0.5)).toBeCloseTo(1 + 2 * 0.5, 9);
  });
});

describe('gibbsSampleMvn', () => {
  it('reproduces a reference sweep-by-sweep trajectory bit-for-bit for a fixed seed', () => {
    const cases = casesFor('gibbsSampleMvn');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 6);
      const result = gibbsSampleMvn(
        rng,
        { mean: c['mean'] as number[], cov: c['cov'] as number[][] },
        c['initial'] as number[],
        c['nSweeps'] as number,
      );
      const expected = c['expected'] as { states: number[][]; updates: { index: number; z: number[] }[] };
      expect(result.states).toHaveLength(expected.states.length);
      result.states.forEach((s, i) => {
        expect(closeTo(s[0]!, expected.states[i]![0]!)).toBe(true);
        expect(closeTo(s[1]!, expected.states[i]![1]!)).toBe(true);
      });
      expect(result.updates).toHaveLength(expected.updates.length);
      result.updates.forEach((u, i) => {
        expect(u.index).toBe(expected.updates[i]!.index);
        expect(closeTo(u.z[0]!, expected.updates[i]!.z[0]!)).toBe(true);
        expect(closeTo(u.z[1]!, expected.updates[i]!.z[1]!)).toBe(true);
      });
    }
  });

  it('recovers the target mean and covariance from a long run', () => {
    const mean = [0, 0];
    const cov = [
      [1, 0.8],
      [0.8, 1],
    ];
    const rng = pcg32(99, 20);
    const result = gibbsSampleMvn(rng, { mean, cov }, [4, -4], 20000);
    const burnIn = 500;
    const kept = result.states.slice(burnIn);
    const m0 = kept.reduce((s, z) => s + z[0]!, 0) / kept.length;
    const m1 = kept.reduce((s, z) => s + z[1]!, 0) / kept.length;
    const v0 = kept.reduce((s, z) => s + (z[0]! - m0) ** 2, 0) / kept.length;
    const cov01 = kept.reduce((s, z) => s + (z[0]! - m0) * (z[1]! - m1), 0) / kept.length;
    expect(Math.abs(m0)).toBeLessThan(0.1);
    expect(Math.abs(m1)).toBeLessThan(0.1);
    expect(Math.abs(v0 - 1)).toBeLessThan(0.15);
    expect(Math.abs(cov01 - 0.8)).toBeLessThan(0.15);
  });

  it('strong correlation makes progress off a diagonal starting point slow, because the conditional variance there is tiny', () => {
    // Starting on the ridge [5, 5]: the conditional mean for either coordinate given
    // the other is `rho * other`, so at rho = 0.99 each step barely leaves 5 (variance
    // 1 - rho^2 = 0.0199), while at rho = 0.01 each step jumps straight towards 0.
    const initial = [5, 5];
    const correlated = { mean: [0, 0], cov: [[1, 0.99], [0.99, 1]] };
    const independent = { mean: [0, 0], cov: [[1, 0.01], [0.01, 1]] };

    const rngA = pcg32(1, 30);
    const chainA = gibbsSampleMvn(rngA, correlated, initial, 3);
    const distA = Math.hypot(chainA.states[3]![0]! - initial[0]!, chainA.states[3]![1]! - initial[1]!);

    const rngB = pcg32(1, 30);
    const chainB = gibbsSampleMvn(rngB, independent, initial, 3);
    const distB = Math.hypot(chainB.states[3]![0]! - initial[0]!, chainB.states[3]![1]! - initial[1]!);

    expect(distA).toBeLessThan(distB);
  });
});
