import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { evaluateFactor, factorById, type FactorGraph } from './factorGraph.js';

interface Case {
  readonly fn: string;
  readonly graph: FactorGraph;
  readonly factorId: string;
  readonly assignment: Readonly<Record<string, number>>;
  readonly expected: number;
}

const fixture = loadFixture<{ cases: Case[] }>('factorGraph');

describe('evaluateFactor', () => {
  it('looks up the flattened row-major table entry for an assignment, matching an independent numpy encoder', () => {
    for (const c of fixture.cases) {
      const factor = factorById(c.graph, c.factorId);
      expect(evaluateFactor(factor, c.graph.variables, c.assignment)).toBeCloseTo(c.expected, 9);
    }
  });
});
