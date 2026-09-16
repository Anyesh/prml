/**
 * The max-sum algorithm, PRML 8.4.5, generalised from the book's chain (Viterbi) case to
 * any tree-structured factor graph: a single collect pass towards `root` in the log
 * domain (so a long chain of small probabilities never underflows), then a backtrack
 * outward that reads off the whole MAP configuration from tables recorded during collect,
 * the same two-stage shape 8.5.3's chain example uses, just not restricted to a chain.
 */

import { isFactorNode, tableStrides, treeOrderFrom, variableById, type FactorGraph } from './factorGraph.js';
import type { Message } from './sumProduct.js';

export interface Backtrack {
  readonly node: string;
  readonly value: number;
}

export interface MaxSumResult {
  readonly root: string;
  /** Invariant: leaves-to-root only. Recovering the MAP needs a backtrack, not a second, root-to-leaves, message pass. */
  readonly schedule: readonly Message[];
  readonly maxLogValue: number;
  readonly assignment: Readonly<Record<string, number>>;
  /** Decision order: `root` first, then outward, matching how the widget unwinds the backtrack. */
  readonly backtrack: readonly Backtrack[];
}

interface FactorBackpointer {
  readonly factorId: string;
  readonly to: string;
  readonly others: readonly string[];
  /** `argmaxOthers[toValue]` is the achieving value of each entry of `others`, in the same order. */
  readonly argmaxOthers: readonly (readonly number[])[];
}

function forEachCombo(sizes: readonly number[], visit: (values: readonly number[]) => void): void {
  const values = new Array(sizes.length).fill(0);
  const total = sizes.reduce((a, b) => a * b, 1);
  for (let flat = 0; flat < total; flat++) {
    let rem = flat;
    for (let i = sizes.length - 1; i >= 0; i--) {
      values[i] = rem % sizes[i]!;
      rem = Math.floor(rem / sizes[i]!);
    }
    visit(values);
  }
}

/**
 * PRML 8.93, generalised to any arity: `logMessage[t] = max over the other scope
 * variables' joint values of (log-table + sum of their incoming log messages)`, with the
 * achieving combination recorded per value of `t` so the backtrack can read it straight
 * back rather than re-searching.
 */
function factorToVariableMessage(
  graph: FactorGraph,
  factorId: string,
  to: string,
  received: ReadonlyMap<string, readonly number[]>,
): { values: number[]; backpointer: FactorBackpointer } {
  const factor = graph.factors.find((f) => f.id === factorId)!;
  const scopeSizes = factor.scope.map((id) => variableById(graph, id).states);
  const strides = tableStrides(scopeSizes);
  const toIndex = factor.scope.indexOf(to);
  const toStates = variableById(graph, to).states;
  const others = factor.scope.filter((v) => v !== to);
  const otherSizes = others.map((id) => variableById(graph, id).states);

  const values = new Array(toStates).fill(-Infinity);
  const argmaxOthers: number[][] = Array.from({ length: toStates }, () => new Array(others.length).fill(0));

  for (let t = 0; t < toStates; t++) {
    forEachCombo(otherSizes, (otherValues) => {
      let index = t * strides[toIndex]!;
      let incoming = 0;
      others.forEach((v, i) => {
        const value = otherValues[i]!;
        index += value * strides[factor.scope.indexOf(v)]!;
        incoming += (received.get(v) ?? [])[value] ?? 0;
      });
      const candidate = Math.log(factor.table[index]!) + incoming;
      if (candidate > values[t]!) {
        values[t] = candidate;
        argmaxOthers[t] = [...otherValues];
      }
    });
  }

  return { values, backpointer: { factorId, to, others, argmaxOthers } };
}

/** A variable's outgoing log-message is the sum of every other incoming log-message (product becomes sum in the log domain). */
function variableToFactorMessage(
  graph: FactorGraph,
  variableId: string,
  to: string,
  received: ReadonlyMap<string, readonly number[]>,
): number[] {
  const states = variableById(graph, variableId).states;
  const result = new Array(states).fill(0);
  for (const [from, values] of received) {
    if (from === to) continue;
    for (let s = 0; s < states; s++) result[s] += values[s]!;
  }
  return result;
}

export function maxSum(graph: FactorGraph, root: string): MaxSumResult {
  const { order, parent, children } = treeOrderFrom(graph, root);
  const received = new Map<string, Map<string, readonly number[]>>();
  for (const node of order) received.set(node, new Map());
  const schedule: Message[] = [];
  const backpointers: FactorBackpointer[] = [];

  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i]!;
    const p = parent.get(node)!;
    if (p === null) continue;

    let values: number[];
    if (isFactorNode(graph, node)) {
      const result = factorToVariableMessage(graph, node, p, received.get(node)!);
      backpointers.push(result.backpointer);
      values = result.values;
    } else {
      values = variableToFactorMessage(graph, node, p, received.get(node)!);
    }
    schedule.push({ from: node, to: p, values });
    received.get(p)!.set(node, values);
  }

  const rootStates = variableById(graph, root).states;
  const rootLog = new Array(rootStates).fill(0);
  for (const values of received.get(root)!.values()) {
    for (let s = 0; s < rootStates; s++) rootLog[s] += values[s]!;
  }
  let rootValue = 0;
  let maxLogValue = -Infinity;
  for (let s = 0; s < rootStates; s++) {
    if (rootLog[s]! > maxLogValue) {
      maxLogValue = rootLog[s]!;
      rootValue = s;
    }
  }

  const assignment: Record<string, number> = { [root]: rootValue };
  const backtrack: Backtrack[] = [{ node: root, value: rootValue }];
  const frontier: string[] = [root];

  while (frontier.length > 0) {
    const node = frontier.shift()!;
    for (const child of children.get(node) ?? []) {
      if (isFactorNode(graph, child)) {
        const bp = backpointers.find((b) => b.factorId === child && b.to === node)!;
        const otherValues = bp.argmaxOthers[assignment[node]!]!;
        bp.others.forEach((v, i) => {
          assignment[v] = otherValues[i]!;
          backtrack.push({ node: v, value: otherValues[i]! });
        });
      }
      frontier.push(child);
    }
  }

  return { root, schedule, maxLogValue, assignment, backtrack };
}
