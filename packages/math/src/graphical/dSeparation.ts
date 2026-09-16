/**
 * D-separation over a directed acyclic graph, PRML 8.2. `dSeparated` answers the yes/no
 * question with the linear-time reachability algorithm (Koller & Friedman's "Reachable",
 * not naive path enumeration, which is exponential); `classifyPath` and `undirectedPaths`
 * exist alongside it purely to drive the explorer widget, which needs to show *which*
 * path blocks and *why*, not just the final verdict.
 */

export interface DirectedGraph {
  readonly nodes: readonly string[];
  /** `[parent, child]` pairs. */
  readonly edges: readonly (readonly [string, string])[];
}

export function parentsOf(graph: DirectedGraph, node: string): string[] {
  return graph.edges.filter(([, child]) => child === node).map(([parent]) => parent);
}

export function childrenOf(graph: DirectedGraph, node: string): string[] {
  return graph.edges.filter(([parent]) => parent === node).map(([, child]) => child);
}

/** Every ancestor of any node in `nodes`, not including the nodes themselves. */
export function ancestorsOf(graph: DirectedGraph, nodes: readonly string[]): Set<string> {
  const result = new Set<string>();
  const stack = [...nodes];
  while (stack.length > 0) {
    const node = stack.pop()!;
    for (const parent of parentsOf(graph, node)) {
      if (!result.has(parent)) {
        result.add(parent);
        stack.push(parent);
      }
    }
  }
  return result;
}

type Direction = 'up' | 'down';

/**
 * Koller & Friedman, *Probabilistic Graphical Models*, algorithm 3.1. A trail from `x`
 * can pass through a non-collider only while travelling straight through it unobserved,
 * and through a collider only when it or a descendant is observed; tracking the arrival
 * direction at each node (`up` from a child, `down` from a parent) is what lets a single
 * traversal enforce both rules without ever materialising a path.
 */
export function dSeparated(graph: DirectedGraph, x: string, y: string, observed: readonly string[]): boolean {
  const z = new Set(observed);
  const ancestorsOfZ = ancestorsOf(graph, observed);
  for (const o of observed) ancestorsOfZ.add(o);

  const visited = new Set<string>();
  const reachable = new Set<string>();
  const stack: [string, Direction][] = [[x, 'up']];

  while (stack.length > 0) {
    const [node, direction] = stack.pop()!;
    const key = `${node}:${direction}`;
    if (visited.has(key)) continue;
    visited.add(key);

    if (!z.has(node)) reachable.add(node);

    if (direction === 'up' && !z.has(node)) {
      for (const parent of parentsOf(graph, node)) stack.push([parent, 'up']);
      for (const child of childrenOf(graph, node)) stack.push([child, 'down']);
    } else if (direction === 'down') {
      if (!z.has(node)) {
        for (const child of childrenOf(graph, node)) stack.push([child, 'down']);
      }
      if (ancestorsOfZ.has(node)) {
        for (const parent of parentsOf(graph, node)) stack.push([parent, 'up']);
      }
    }
  }

  return !reachable.has(y);
}

export type NodeRole = 'tail-to-tail' | 'head-to-tail' | 'head-to-head';

export interface PathStep {
  readonly node: string;
  readonly role: NodeRole;
  /** Whether this node alone blocks the path, given the observed set. */
  readonly blocking: boolean;
}

export interface ClassifiedPath {
  readonly nodes: readonly string[];
  /** True iff any interior node blocks, which is the same as the whole path blocking. */
  readonly blocked: boolean;
  readonly path: readonly PathStep[];
}

function isEdge(graph: DirectedGraph, from: string, to: string): boolean {
  return graph.edges.some(([p, c]) => p === from && c === to);
}

/**
 * All simple paths between `x` and `y` treating edges as undirected, for a small graph
 * (the explorer's canonical examples and worked cases never exceed a handful of nodes;
 * this is not meant for graphs where path count could blow up).
 */
export function undirectedPaths(graph: DirectedGraph, x: string, y: string): readonly (readonly string[])[] {
  const neighbours = new Map<string, string[]>();
  for (const node of graph.nodes) neighbours.set(node, []);
  for (const [a, b] of graph.edges) {
    neighbours.get(a)!.push(b);
    neighbours.get(b)!.push(a);
  }

  const results: string[][] = [];
  const visiting = new Set<string>([x]);
  const path: string[] = [x];

  function dfs(current: string) {
    if (current === y) {
      results.push([...path]);
      return;
    }
    for (const next of neighbours.get(current) ?? []) {
      if (visiting.has(next)) continue;
      visiting.add(next);
      path.push(next);
      dfs(next);
      path.pop();
      visiting.delete(next);
    }
  }

  dfs(x);
  return results;
}

/**
 * PRML 8.2, page 373-374: a tail-to-tail or head-to-tail node blocks the path iff it is
 * observed; a head-to-head node (a collider) blocks unless it or one of its descendants
 * is observed, the explaining-away reversal.
 */
export function classifyPath(graph: DirectedGraph, path: readonly string[], observed: readonly string[]): ClassifiedPath {
  const z = new Set(observed);
  const ancestorsOfZ = ancestorsOf(graph, observed);
  for (const o of observed) ancestorsOfZ.add(o);

  const steps: PathStep[] = [];
  for (let i = 1; i < path.length - 1; i++) {
    const prev = path[i - 1]!;
    const node = path[i]!;
    const next = path[i + 1]!;

    const prevIsParent = isEdge(graph, prev, node);
    const nextIsParent = isEdge(graph, next, node);
    const nodeToPrev = isEdge(graph, node, prev);
    const nodeToNext = isEdge(graph, node, next);

    let role: NodeRole;
    let blocking: boolean;
    if (prevIsParent && nextIsParent) {
      role = 'head-to-head';
      blocking = !z.has(node) && !ancestorsOfZ.has(node);
    } else if (nodeToPrev && nodeToNext) {
      role = 'tail-to-tail';
      blocking = z.has(node);
    } else {
      role = 'head-to-tail';
      blocking = z.has(node);
    }
    steps.push({ node, role, blocking });
  }

  return { nodes: path, blocked: steps.some((s) => s.blocking), path: steps };
}
