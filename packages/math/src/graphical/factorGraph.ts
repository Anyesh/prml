/**
 * Factor graphs, PRML 8.4.3: a bipartite graph of variable nodes and factor nodes, each
 * factor's table one term of a joint that factorises as their product. The sum-product
 * and max-sum algorithms in this directory both operate on this representation, and
 * chapter 13 builds its scaled forward-backward and Viterbi on the same `sumProduct` and
 * `maxSum` by expressing a hidden Markov chain as the chain factor graph below.
 */

export interface Variable {
  readonly id: string;
  /** Number of discrete states, valued 0..states-1. */
  readonly states: number;
}

export interface Factor {
  readonly id: string;
  /** Variable ids this factor depends on, in the order `table` is indexed. */
  readonly scope: readonly string[];
  /**
   * Flattened row-major over `scope`, the last scope variable fastest (numpy's default
   * C order), length `product(states of scope variables)`.
   */
  readonly table: readonly number[];
}

export interface FactorGraph {
  readonly variables: readonly Variable[];
  readonly factors: readonly Factor[];
}

function findVariable(variables: readonly Variable[], id: string): Variable {
  const v = variables.find((x) => x.id === id);
  if (!v) throw new Error(`@prml/math: no variable "${id}" in this factor graph`);
  return v;
}

export function variableById(graph: FactorGraph, id: string): Variable {
  return findVariable(graph.variables, id);
}

export function factorById(graph: FactorGraph, id: string): Factor {
  const f = graph.factors.find((x) => x.id === id);
  if (!f) throw new Error(`@prml/math: no factor "${id}" in this factor graph`);
  return f;
}

/** Strides for row-major indexing over `sizes`, last dimension fastest. */
export function tableStrides(sizes: readonly number[]): number[] {
  const strides = new Array(sizes.length).fill(1);
  for (let i = sizes.length - 2; i >= 0; i--) strides[i] = strides[i + 1]! * sizes[i + 1]!;
  return strides;
}

/** A factor's table entry for a full assignment to every variable in its scope. */
export function evaluateFactor(
  factor: Factor,
  variables: readonly Variable[],
  assignment: Readonly<Record<string, number>>,
): number {
  const sizes = factor.scope.map((id) => findVariable(variables, id).states);
  const strides = tableStrides(sizes);
  let index = 0;
  for (let i = 0; i < factor.scope.length; i++) {
    const value = assignment[factor.scope[i]!];
    if (value === undefined) {
      throw new Error(`@prml/math: assignment is missing scope variable "${factor.scope[i]}"`);
    }
    index += value * strides[i]!;
  }
  return factor.table[index]!;
}

/**
 * Every neighbour id of a node in the bipartite variable/factor graph: a variable's
 * neighbours are the factors whose scope contains it, a factor's neighbours are its
 * scope itself.
 */
export function neighborsOf(graph: FactorGraph, nodeId: string): readonly string[] {
  const factor = graph.factors.find((f) => f.id === nodeId);
  if (factor) return factor.scope;
  const isVariable = graph.variables.some((v) => v.id === nodeId);
  if (!isVariable) throw new Error(`@prml/math: no node "${nodeId}" in this factor graph`);
  return graph.factors.filter((f) => f.scope.includes(nodeId)).map((f) => f.id);
}

export function isFactorNode(graph: FactorGraph, nodeId: string): boolean {
  return graph.factors.some((f) => f.id === nodeId);
}

export interface TreeOrder {
  /** BFS order from the root: `order[0]` is the root, every parent precedes its children. */
  readonly order: readonly string[];
  readonly parent: ReadonlyMap<string, string | null>;
  readonly children: ReadonlyMap<string, readonly string[]>;
}

/**
 * BFS spanning order from `root` over the whole bipartite graph. Throws if the graph is
 * not a tree (a node reached twice means a cycle) or not connected (a node never
 * reached), because both sum-product and max-sum assume exactness that only holds on a
 * tree-structured factor graph.
 */
export function treeOrderFrom(graph: FactorGraph, root: string): TreeOrder {
  const allNodes = new Set<string>([...graph.variables.map((v) => v.id), ...graph.factors.map((f) => f.id)]);
  if (!allNodes.has(root)) throw new Error(`@prml/math: root "${root}" is not a node in this factor graph`);

  const parent = new Map<string, string | null>([[root, null]]);
  const children = new Map<string, string[]>();
  for (const node of allNodes) children.set(node, []);
  const order: string[] = [root];
  const queue = [root];

  while (queue.length > 0) {
    const node = queue.shift()!;
    for (const neighbor of neighborsOf(graph, node)) {
      if (parent.has(neighbor)) {
        if (parent.get(node) !== neighbor) {
          throw new Error(`@prml/math: factor graph has a cycle through "${neighbor}"; sum-product and max-sum require a tree`);
        }
        continue;
      }
      parent.set(neighbor, node);
      children.get(node)!.push(neighbor);
      order.push(neighbor);
      queue.push(neighbor);
    }
  }

  if (order.length !== allNodes.size) {
    throw new Error('@prml/math: factor graph is disconnected; sum-product and max-sum require a single tree');
  }

  return { order, parent, children };
}
