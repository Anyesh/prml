import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { maxSum } from './maxSum.js';
import type { FactorGraph } from './factorGraph.js';

interface Case {
  readonly fn: string;
  readonly graph: FactorGraph;
  readonly root: string;
  readonly expectedAssignment: Readonly<Record<string, number>>;
  readonly expectedMaxLogValue: number;
}

const fixture = loadFixture<{ cases: Case[] }>('maxSum');

describe('maxSum', () => {
  it('recovers the exact MAP configuration and its log value against a brute-force argmax over the joint', () => {
    for (const c of fixture.cases) {
      const result = maxSum(c.graph, c.root);
      expect(result.assignment).toEqual(c.expectedAssignment);
      expect(result.maxLogValue).toBeCloseTo(c.expectedMaxLogValue, 9);
    }
  });

  it('backtracks every variable exactly once, ending at values consistent with the reported assignment', () => {
    for (const c of fixture.cases) {
      const result = maxSum(c.graph, c.root);
      expect(result.backtrack.map((b) => b.node).sort()).toEqual(c.graph.variables.map((v) => v.id).sort());
      for (const step of result.backtrack) {
        expect(step.value).toBe(result.assignment[step.node]);
      }
    }
  });
});
