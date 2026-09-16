/**
 * The sum-product algorithm, PRML 8.4.4, general over any tree-structured factor graph
 * rather than specialised to a chain. Chapter 13's scaled forward-backward is this same
 * algorithm run on the chain factor graph a hidden Markov model reduces to; if this
 * implementation only worked on chains, that equivalence would be a claim rather than a
 * fact a caller could check.
 */

import { isFactorNode, tableStrides, treeOrderFrom, variableById, type FactorGraph } from './factorGraph.js';

export interface Message {
  readonly from: string;
  readonly to: string;
  /** Indexed by the *target*'s states: `to`'s states if `to` is a variable, else the sender's. */
  readonly values: readonly number[];
}

export interface SumProductResult {
  /**
   * Invariant: collect phase (leaves to root) followed by distribute phase (root to
   * leaves), so a variable's marginal is exact only once every message that will ever
   * reach it has arrived, which is after the whole schedule has played, not partway.
   */
  readonly schedule: readonly Message[];
  /** Every variable's exact marginal, normalised to sum to 1. */
  readonly marginals: Readonly<Record<string, readonly number[]>>;
}

function productOf(vectors: readonly (readonly number[])[], length: number): number[] {
  const result = new Array(length).fill(1);
  for (const v of vectors) for (let i = 0; i < length; i++) result[i] *= v[i]!;
  return result;
}

/** All combinations of `sizes`, row-major (last fastest), each yielded as a value array. */
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
 * A message out of a factor node towards `to`, PRML 8.66: the factor's table, weighted
 * by every other incoming variable-to-factor message, summed over every other variable.
 */
function factorToVariableMessage(
  graph: FactorGraph,
  factorId: string,
  to: string,
  received: ReadonlyMap<string, readonly number[]>,
): number[] {
  const factor = graph.factors.find((f) => f.id === factorId)!;
  const scopeSizes = factor.scope.map((id) => variableById(graph, id).states);
  const strides = tableStrides(scopeSizes);
  const toIndex = factor.scope.indexOf(to);
  const toStates = variableById(graph, to).states;
  const others = factor.scope.filter((v) => v !== to);
  const otherSizes = others.map((id) => variableById(graph, id).states);

  const result = new Array(toStates).fill(0);
  for (let t = 0; t < toStates; t++) {
    let sum = 0;
    forEachCombo(otherSizes, (otherValues) => {
      let index = t * strides[toIndex]!;
      let weight = 1;
      others.forEach((v, i) => {
        const value = otherValues[i]!;
        const scopeIndex = factor.scope.indexOf(v);
        index += value * strides[scopeIndex]!;
        weight *= (received.get(v) ?? [1])[value]!;
      });
      sum += factor.table[index]! * weight;
    });
    result[t] = sum;
  }
  return result;
}

function productOfReceived(states: number, received: ReadonlyMap<string, readonly number[]>, excluding?: string): number[] {
  const incoming = [...received.entries()].filter(([from]) => from !== excluding).map(([, values]) => values);
  return incoming.length === 0 ? new Array(states).fill(1) : productOf(incoming, states);
}

/** PRML 8.69/8.70: a variable's message is the product of every other incoming factor message, or all-ones if it is a leaf. */
function variableToFactorMessage(
  graph: FactorGraph,
  variableId: string,
  to: string,
  received: ReadonlyMap<string, readonly number[]>,
): number[] {
  return productOfReceived(variableById(graph, variableId).states, received, to);
}

function computeMessage(
  graph: FactorGraph,
  from: string,
  to: string,
  receivedByFrom: ReadonlyMap<string, readonly number[]>,
): number[] {
  return isFactorNode(graph, from)
    ? factorToVariableMessage(graph, from, to, receivedByFrom)
    : variableToFactorMessage(graph, from, to, receivedByFrom);
}

export function sumProduct(graph: FactorGraph, root: string): SumProductResult {
  const { order, parent, children } = treeOrderFrom(graph, root);
  const received = new Map<string, Map<string, readonly number[]>>();
  for (const node of order) received.set(node, new Map());
  const schedule: Message[] = [];

  function send(from: string, to: string) {
    const values = computeMessage(graph, from, to, received.get(from)!);
    schedule.push({ from, to, values });
    received.get(to)!.set(from, values);
  }

  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i]!;
    const p = parent.get(node)!;
    if (p !== null) send(node, p);
  }

  for (const node of order) {
    for (const child of children.get(node) ?? []) send(node, child);
  }

  const marginals: Record<string, readonly number[]> = {};
  for (const variable of graph.variables) {
    const raw = productOfReceived(variable.states, received.get(variable.id)!);
    const total = raw.reduce((a, b) => a + b, 0);
    marginals[variable.id] = raw.map((v) => v / total);
  }

  return { schedule, marginals };
}

/**
 * The marginal implied by whichever messages in `messages` have already arrived at
 * `variableId`: a factor node not yet heard from contributes a uniform (all-ones)
 * message, so this converges to the exact marginal as the schedule plays out and lets a
 * widget show a "current belief" mid-animation.
 */
export function marginalFromMessages(graph: FactorGraph, variableId: string, messages: readonly Message[]): readonly number[] {
  const states = variableById(graph, variableId).states;
  const received = new Map<string, readonly number[]>();
  for (const m of messages) if (m.to === variableId) received.set(m.from, m.values);
  const neighbors = graph.factors.filter((f) => f.scope.includes(variableId)).map((f) => f.id);
  const vectors = neighbors.map((n) => received.get(n) ?? new Array(states).fill(1));
  const raw = productOf(vectors, states);
  const total = raw.reduce((a, b) => a + b, 0);
  return total === 0 ? raw : raw.map((v) => v / total);
}
