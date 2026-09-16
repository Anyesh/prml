import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { classifyPath, dSeparated, type DirectedGraph } from './dSeparation.js';

interface DSeparatedCase {
  readonly graph: string;
  readonly nodes: readonly string[];
  readonly edges: readonly (readonly [string, string])[];
  readonly x: string;
  readonly y: string;
  readonly observed: readonly string[];
  readonly expected: boolean;
}

interface ClassifyPathCase {
  readonly graph: string;
  readonly nodes: readonly string[];
  readonly edges: readonly (readonly [string, string])[];
  readonly observed: readonly string[];
  readonly path: readonly string[];
  readonly expectedBlocked: boolean;
  readonly expectedSteps: readonly { readonly node: string; readonly role: string; readonly blocking: boolean }[];
}

const fixture = loadFixture<{ dSeparated: DSeparatedCase[]; classifyPath: ClassifyPathCase[] }>('dSeparation');

function toGraph(nodes: readonly string[], edges: readonly (readonly [string, string])[]): DirectedGraph {
  return { nodes, edges };
}

describe('dSeparated', () => {
  it('agrees with the exhaustive networkx oracle over every pair and observed subset on four graphs', () => {
    for (const c of fixture.dSeparated) {
      const graph = toGraph(c.nodes, c.edges);
      expect(dSeparated(graph, c.x, c.y, c.observed)).toBe(c.expected);
    }
  });
});

describe('classifyPath', () => {
  it('assigns the same per-node role and blocking verdict as the independent Python classifier', () => {
    for (const c of fixture.classifyPath) {
      const graph = toGraph(c.nodes, c.edges);
      const result = classifyPath(graph, c.path, c.observed);
      expect(result.blocked).toBe(c.expectedBlocked);
      expect(result.path.map((s) => ({ node: s.node, role: s.role, blocking: s.blocking }))).toEqual(
        c.expectedSteps,
      );
    }
  });
});
