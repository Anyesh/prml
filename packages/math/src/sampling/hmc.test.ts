import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { solve } from '../linalg/decompose.js';
import { vecSub } from '../linalg/core.js';
import { hamiltonian, hmcChain, kineticEnergy, leapfrogStep } from './hmc.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_hmc');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

const MEAN = [0.5, -0.2];
const COV = [
  [1.5, 0.9],
  [0.9, 1.0],
];

function energyFn(z: readonly number[]): number {
  const d = vecSub(z, MEAN);
  const solved = solve(COV, d);
  let sum = 0;
  for (let i = 0; i < d.length; i++) sum += d[i]! * solved[i]!;
  return 0.5 * sum;
}

function gradEnergyFn(z: readonly number[]): number[] {
  return solve(COV, vecSub(z, MEAN));
}

describe('kineticEnergy / hamiltonian', () => {
  it('matches ||r||^2/2 plus the potential exactly (PRML 11.56-11.57)', () => {
    for (const c of casesFor('hamiltonian')) {
      const z = c['z'] as number[];
      const r = c['r'] as number[];
      expect(closeTo(hamiltonian(z, r, energyFn), c['expected'] as number)).toBe(true);
      expect(closeTo(kineticEnergy(r), 0.5 * r.reduce((s, x) => s + x * x, 0))).toBe(true);
    }
  });
});

describe('leapfrogStep', () => {
  it('matches an independent numpy leapfrog step from a fixed (z, r), with no RNG involved', () => {
    const cases = casesFor('leapfrogStep');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const result = leapfrogStep({ z: c['z'] as number[], r: c['r'] as number[] }, gradEnergyFn, c['epsilon'] as number);
      const expected = c['expected'] as { z: number[]; r: number[] };
      result.z.forEach((v, i) => expect(closeTo(v, expected.z[i]!)).toBe(true));
      result.r.forEach((v, i) => expect(closeTo(v, expected.r[i]!)).toBe(true));
    }
  });
});

describe('hmcChain', () => {
  it('reproduces a reference chain bit-for-bit for a fixed seed', () => {
    const cases = casesFor('hmcChain');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 14);
      const result = hmcChain(rng, [0, 0], c['nSteps'] as number, {
        energyFn,
        gradEnergyFn,
        epsilon: c['epsilon'] as number,
        l: c['l'] as number,
      });
      const expected = c['expected'] as { states: number[][]; accepted: boolean[]; acceptanceRate: number };
      expect(result.states).toHaveLength(expected.states.length);
      result.states.forEach((s, i) => {
        expect(closeTo(s[0]!, expected.states[i]![0]!)).toBe(true);
        expect(closeTo(s[1]!, expected.states[i]![1]!)).toBe(true);
      });
      expect(result.accepted).toEqual(expected.accepted);
      expect(closeTo(result.acceptanceRate, expected.acceptanceRate)).toBe(true);
    }
  });

  it('recovers the target mean and covariance from a long run', () => {
    const rng = pcg32(555, 40);
    const result = hmcChain(rng, [0, 0], 4000, { energyFn, gradEnergyFn, epsilon: 0.2, l: 8 });
    const kept = result.states.slice(200);
    const m0 = kept.reduce((s, z) => s + z[0]!, 0) / kept.length;
    const m1 = kept.reduce((s, z) => s + z[1]!, 0) / kept.length;
    expect(Math.abs(m0 - MEAN[0]!)).toBeLessThan(0.15);
    expect(Math.abs(m1 - MEAN[1]!)).toBeLessThan(0.15);
    expect(result.acceptanceRate).toBeGreaterThan(0.5);
  });

  it('for a matched gradient/likelihood-evaluation budget, HMC covers more mean-squared distance than a random-walk Metropolis chain, averaged over many independent runs', () => {
    // PRML 11.5.2's closing comparison: HMC needs O(sigma_max/sigma_min) leapfrog
    // evaluations to reach a roughly independent state, against O((sigma_max/sigma_min)^2)
    // for a random walk of matched step size. Both sides here spend exactly 10 * 40 = 400
    // target evaluations; a single run is noisy, so this averages 40 independent runs.
    const l = 10;
    const nHmcSteps = 40;
    const stepSize = 0.5;

    let hmcSumSq = 0;
    let walkSumSq = 0;
    const runs = 40;
    for (let run = 0; run < runs; run++) {
      const hmc = hmcChain(pcg32(1000 + run, 41), [0, 0], nHmcSteps, { energyFn, gradEnergyFn, epsilon: stepSize / l, l });
      const hmcEnd = hmc.states[hmc.states.length - 1]!;
      hmcSumSq += hmcEnd[0]! ** 2 + hmcEnd[1]! ** 2;

      const walkRng = pcg32(1000 + run, 42);
      let z = [0, 0];
      for (let i = 0; i < nHmcSteps * l; i++) {
        const candidate = [z[0]! + stepSize * (walkRng.next() - 0.5), z[1]! + stepSize * (walkRng.next() - 0.5)];
        const logRatio = -energyFn(candidate) + energyFn(z);
        if (walkRng.next() < Math.min(1, Math.exp(logRatio))) z = candidate;
      }
      walkSumSq += z[0]! ** 2 + z[1]! ** 2;
    }

    expect(hmcSumSq / runs).toBeGreaterThan(walkSumSq / runs);
  });
});
