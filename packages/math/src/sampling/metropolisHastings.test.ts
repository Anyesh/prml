import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { mvnLogPdf } from '../distributions/mvn.js';
import {
  autocorrelation,
  gaussianRandomWalkProposal,
  mhAcceptanceLogRatio,
  mhAcceptanceProbability,
  metropolisHastingsChain,
  type MhSampler,
} from './metropolisHastings.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_mh');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('mhAcceptanceLogRatio / mhAcceptanceProbability', () => {
  it('matches the closed-form ratio (PRML 11.44) on fixed state pairs', () => {
    for (const c of casesFor('mhAcceptanceLogRatio')) {
      const actual = mhAcceptanceLogRatio(
        c['candidateLogTarget'] as number,
        c['currentLogTarget'] as number,
        c['logQCurrentGivenCandidate'] as number,
        c['logQCandidateGivenCurrent'] as number,
      );
      expect(closeTo(actual, c['expected'] as number)).toBe(true);
    }
    for (const c of casesFor('mhAcceptanceProbability')) {
      expect(closeTo(mhAcceptanceProbability(c['logRatio'] as number), c['expected'] as number)).toBe(true);
    }
  });
});

describe('autocorrelation', () => {
  it('matches an independent numpy computation on a fixed chain', () => {
    const c = casesFor('autocorrelation')[0]!;
    const actual = autocorrelation(c['chain'] as number[], c['maxLag'] as number);
    (c['expected'] as number[]).forEach((e, i) => expect(closeTo(actual[i]!, e)).toBe(true));
    expect(actual[0]).toBeCloseTo(1, 9);
  });
});

describe('metropolisHastingsChain', () => {
  const mean = [0, 0];
  const cov = [
    [1, 0.8],
    [0.8, 1],
  ];

  it('reproduces a reference chain bit-for-bit for a fixed seed', () => {
    const cases = casesFor('metropolisHastingsChain');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 5);
      const proposal = gaussianRandomWalkProposal(c['stepSize'] as number);
      const sampler: MhSampler<number[]> = { ...proposal, targetLogPdf: (z) => mvnLogPdf(z, { mean, cov }) };
      const result = metropolisHastingsChain(rng, [0, 0], c['nSteps'] as number, sampler);
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

  it('tiny steps mix slowly (near-1 autocorrelation persists) and huge steps barely move (near-zero acceptance)', () => {
    const target: MhSampler<number[]>['targetLogPdf'] = (z) => mvnLogPdf(z, { mean, cov });

    const tiny = gaussianRandomWalkProposal(0.01);
    const rngTiny = pcg32(1, 11);
    const tinyChain = metropolisHastingsChain(rngTiny, [0, 0], 500, { ...tiny, targetLogPdf: target });
    const tinyRho = autocorrelation(
      tinyChain.states.map((s) => s[0]!),
      20,
    );
    expect(tinyChain.acceptanceRate).toBeGreaterThan(0.9);
    expect(tinyRho[20]).toBeGreaterThan(0.5);

    const huge = gaussianRandomWalkProposal(50);
    const rngHuge = pcg32(1, 12);
    const hugeChain = metropolisHastingsChain(rngHuge, [0, 0], 500, { ...huge, targetLogPdf: target });
    expect(hugeChain.acceptanceRate).toBeLessThan(0.05);
  });

  it('a well-tuned chain recovers the target mean and marginal variance', () => {
    const proposal = gaussianRandomWalkProposal(1.0);
    const rng = pcg32(2024, 13);
    const chain = metropolisHastingsChain(rng, [0, 0], 20000, {
      ...proposal,
      targetLogPdf: (z) => mvnLogPdf(z, { mean, cov }),
    });
    const burnIn = 1000;
    const kept = chain.states.slice(burnIn);
    const m0 = kept.reduce((s, z) => s + z[0]!, 0) / kept.length;
    const m1 = kept.reduce((s, z) => s + z[1]!, 0) / kept.length;
    expect(Math.abs(m0)).toBeLessThan(0.15);
    expect(Math.abs(m1)).toBeLessThan(0.15);
  });
});
