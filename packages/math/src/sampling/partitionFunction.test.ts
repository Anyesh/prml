import { describe, expect, it } from 'vitest';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { chainedPartitionFunctionRatio, interpolatedEnergy, partitionFunctionRatioEstimate } from './partitionFunction.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_partition');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

function gaussianEnergy(sigma: number): (z: number) => number {
  return (z) => (z * z) / (2 * sigma * sigma);
}

describe('partitionFunctionRatioEstimate', () => {
  it('matches the exact finite-sample sum for a fixed sample set (PRML 11.72)', () => {
    const c = casesFor('partitionFunctionRatioEstimate')[0]!;
    const energyE = gaussianEnergy(c['sigmaE'] as number);
    const energyG = gaussianEnergy(c['sigmaG'] as number);
    const actual = partitionFunctionRatioEstimate(c['samples'] as number[], energyE, energyG);
    expect(closeTo(actual, c['expected'] as number)).toBe(true);
  });

  it('the finite-sample estimate is in the right ballpark of the true closed-form ratio for these two Gaussians', () => {
    const trueRatioCase = casesFor('partitionFunctionRatioEstimate_trueRatio')[0]!;
    const estimateCase = casesFor('partitionFunctionRatioEstimate')[0]!;
    const energyE = gaussianEnergy(estimateCase['sigmaE'] as number);
    const energyG = gaussianEnergy(estimateCase['sigmaG'] as number);
    const actual = partitionFunctionRatioEstimate(estimateCase['samples'] as number[], energyE, energyG);
    const trueRatio = trueRatioCase['expected'] as number;
    expect(Math.abs(actual - trueRatio)).toBeLessThan(1);
  });
});

describe('chainedPartitionFunctionRatio', () => {
  it('multiplies a sequence of ratios exactly (PRML 11.74)', () => {
    for (const c of casesFor('chainedPartitionFunctionRatio')) {
      const actual = chainedPartitionFunctionRatio(c['ratios'] as number[]);
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
  });
});

describe('interpolatedEnergy', () => {
  it('matches (1-alpha)*E1 + alpha*EM exactly (PRML 11.75), reducing to each endpoint energy', () => {
    for (const c of casesFor('interpolatedEnergy')) {
      const energy1 = gaussianEnergy(c['sigmaE'] as number);
      const energyM = gaussianEnergy(c['sigmaG'] as number);
      const interpolated = interpolatedEnergy(c['alpha'] as number, energy1, energyM);
      expect(closeTo(interpolated(c['z'] as number), c['expected'] as number)).toBe(true);
    }
    const energy1 = gaussianEnergy(1.5);
    const energyM = gaussianEnergy(2.0);
    expect(interpolatedEnergy(0, energy1, energyM)(3)).toBeCloseTo(energy1(3), 9);
    expect(interpolatedEnergy(1, energy1, energyM)(3)).toBeCloseTo(energyM(3), 9);
  });
});
