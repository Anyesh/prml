import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { sumProduct } from './sumProduct.js';
import type { FactorGraph } from './factorGraph.js';

interface Case {
  readonly fn: string;
  readonly graph: FactorGraph;
  readonly root: string;
  readonly expectedMarginals: Readonly<Record<string, readonly number[]>>;
  readonly expectedZ: number;
}

const fixture = loadFixture<{ cases: Case[] }>('sumProduct');

describe('sumProduct', () => {
  it('produces exact marginals matching a brute-force enumeration of the joint, for a chain, a tree and a hyperedge factor', () => {
    for (const c of fixture.cases) {
      const result = sumProduct(c.graph, c.root);
      for (const variable of c.graph.variables) {
        const expected = c.expectedMarginals[variable.id]!;
        const actual = result.marginals[variable.id]!;
        expected.forEach((value, i) => expect(actual[i]!).toBeCloseTo(value, 9));
      }
    }
  });

  it('gives the same marginals regardless of which node in the chain is chosen as root', () => {
    const chainCases = fixture.cases.filter(
      (c) => c.graph.variables.some((v) => v.id === 'x1') && c.graph.factors.some((f) => f.id === 'fa'),
    );
    expect(chainCases.length).toBeGreaterThanOrEqual(2);
    const [first, second] = chainCases;
    const resultA = sumProduct(first!.graph, first!.root);
    const resultB = sumProduct(second!.graph, second!.root);
    for (const variable of first!.graph.variables) {
      const a = resultA.marginals[variable.id]!;
      const b = resultB.marginals[variable.id]!;
      a.forEach((value, i) => expect(value).toBeCloseTo(b[i]!, 9));
    }
  });
});
