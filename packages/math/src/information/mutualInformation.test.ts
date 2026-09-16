import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { mutualInformation, conditionalEntropy } from './mutualInformation.js';

interface Case {
  readonly fn: string;
  readonly joint: number[][];
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('information');

function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('mutualInformation', () => {
  it('matches the H[X] + H[Y] - H[X,Y] identity on every golden case', () => {
    const cases = casesFor('mutualInformation');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = mutualInformation(c.joint);
      expect(closeTo(actual, c.expected), `mutualInformation(${JSON.stringify(c.joint)})`).toBe(true);
    }
  });

  it('is zero for an independent joint table', () => {
    const joint = [
      [0.15, 0.35],
      [0.15, 0.35],
    ];
    expect(Math.abs(mutualInformation(joint))).toBeLessThan(1e-12);
  });
});

describe('conditionalEntropy', () => {
  it('matches the H[X,Y] - H[X] identity on every golden case', () => {
    const cases = casesFor('conditionalEntropy');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = conditionalEntropy(c.joint);
      expect(closeTo(actual, c.expected), `conditionalEntropy(${JSON.stringify(c.joint)})`).toBe(true);
    }
  });

  it('is zero when Y is a deterministic function of X', () => {
    const joint = [
      [0.25, 0, 0],
      [0, 0.5, 0],
      [0, 0, 0.25],
    ];
    expect(Math.abs(conditionalEntropy(joint))).toBeLessThan(1e-12);
  });
});
