import type { Mat, Vec } from '../types.js';

/** A regression region's sufficient statistics for SSE, PRML 14.29-14.31: `sumTSq - sumT^2/n`. */
export interface RegressionStat {
  readonly sumT: number;
  readonly sumTSq: number;
}

/** A classification region's per-class counts; `counts[k] / nSamples` is PRML 14.32-14.33's `p_tau_k`. */
export interface ClassificationStat {
  readonly counts: readonly number[];
}

/**
 * `L` is the value a leaf predicts; `S` is whatever statistic that task needs to price
 * collapsing a node during pruning. A regression leaf and a classification leaf are not
 * the same shape of thing, so `S` is not fixed to one shape here: `RegressionStat` for
 * `fitRegressionTree`, `ClassificationStat` for `fitClassificationTree`. Every node, leaf
 * or split, carries its own already-aggregated `stat` for its whole region, computed once
 * during fitting, so `pruneRegressionTree`/`pruneClassificationTree` never need to
 * re-scan raw data or recombine children's statistics: a split node's `stat` already is
 * that combination.
 */
export type CartNode<L, S = RegressionStat> =
  | { readonly kind: 'leaf'; readonly value: L; readonly nSamples: number; readonly stat: S }
  | {
      readonly kind: 'split';
      readonly featureIndex: number;
      readonly threshold: number;
      readonly left: CartNode<L, S>;
      readonly right: CartNode<L, S>;
      readonly nSamples: number;
      readonly stat: S;
    };

export interface CartOptions {
  readonly minLeafSize?: number;
  readonly maxDepth?: number;
}

function resolveOptions(options: CartOptions | undefined): { minLeafSize: number; maxDepth: number } {
  return {
    minLeafSize: options?.minLeafSize ?? 1,
    maxDepth: options?.maxDepth ?? Infinity,
  };
}

/**
 * Exhaustive search over (feature, threshold) candidates in one fixed, deterministic
 * order: feature index ascending, then for each feature the midpoints between
 * consecutive DISTINCT sorted values of that feature, ascending (same convention as
 * `stumps.ts:fitDecisionStump`). `score` is whatever a candidate split should minimise
 * (children SSE for regression, weighted impurity for classification); only a STRICT
 * improvement (`<`, never `<=`) replaces the running best, so when several candidates
 * tie on score, the first one encountered in this order wins - lower feature index
 * first, then lower threshold. `gen_cart.py` walks candidates in the identical order,
 * so the two agree on ties by construction rather than by luck.
 */
function bestSplit(
  X: Mat,
  indices: readonly number[],
  minLeafSize: number,
  score: (leftIndices: number[], rightIndices: number[]) => number,
): { readonly featureIndex: number; readonly threshold: number } | null {
  const numFeatures = X[0]?.length ?? 0;
  let best: { featureIndex: number; threshold: number } | null = null;
  let bestScore = Infinity;

  for (let featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
    const values = Array.from(new Set(indices.map((i) => X[i]![featureIndex]!))).sort((a, b) => a - b);
    for (let i = 0; i + 1 < values.length; i++) {
      const threshold = (values[i]! + values[i + 1]!) / 2;
      const leftIndices = indices.filter((i) => X[i]![featureIndex]! <= threshold);
      const rightIndices = indices.filter((i) => X[i]![featureIndex]! > threshold);
      if (leftIndices.length < minLeafSize || rightIndices.length < minLeafSize) continue;
      const candidateScore = score(leftIndices, rightIndices);
      if (candidateScore < bestScore) {
        bestScore = candidateScore;
        best = { featureIndex, threshold };
      }
    }
  }
  return best;
}

function regressionStat(values: readonly number[], indices: readonly number[]): RegressionStat {
  let sumT = 0;
  let sumTSq = 0;
  for (const i of indices) {
    const v = values[i]!;
    sumT += v;
    sumTSq += v * v;
  }
  return { sumT, sumTSq };
}

/** PRML 14.29-14.31: a region's own SSE, from its sufficient statistics alone. */
function regionSse(stat: RegressionStat, nSamples: number): number {
  return stat.sumTSq - (stat.sumT * stat.sumT) / nSamples;
}

function buildRegressionNode(
  X: Mat,
  t: Vec,
  indices: readonly number[],
  depth: number,
  options: { minLeafSize: number; maxDepth: number },
): CartNode<number, RegressionStat> {
  const nSamples = indices.length;
  const stat = regressionStat(t, indices);
  const leaf = (): CartNode<number, RegressionStat> => ({ kind: 'leaf', value: stat.sumT / nSamples, nSamples, stat });

  if (nSamples < 2 * options.minLeafSize || depth >= options.maxDepth) return leaf();

  const split = bestSplit(X, indices, options.minLeafSize, (leftIndices, rightIndices) => {
    const left = regressionStat(t, leftIndices);
    const right = regressionStat(t, rightIndices);
    return regionSse(left, leftIndices.length) + regionSse(right, rightIndices.length);
  });
  if (!split) return leaf();

  const leftIndices = indices.filter((i) => X[i]![split.featureIndex]! <= split.threshold);
  const rightIndices = indices.filter((i) => X[i]![split.featureIndex]! > split.threshold);
  return {
    kind: 'split',
    featureIndex: split.featureIndex,
    threshold: split.threshold,
    left: buildRegressionNode(X, t, leftIndices, depth + 1, options),
    right: buildRegressionNode(X, t, rightIndices, depth + 1, options),
    nSamples,
    stat,
  };
}

/** PRML 14.29-14.31: leaf value is the mean of `t` in the region, split minimises total child SSE. */
export function fitRegressionTree(X: Mat, t: Vec, options?: CartOptions): CartNode<number, RegressionStat> {
  return buildRegressionNode(X, t, X.map((_, i) => i), 0, resolveOptions(options));
}

export function predictRegressionTree(tree: CartNode<number, RegressionStat>, x: Vec): number {
  let node = tree;
  while (node.kind === 'split') {
    node = x[node.featureIndex]! <= node.threshold ? node.left : node.right;
  }
  return node.value;
}

/** PRML 14.33: `sum_k p_k (1 - p_k)`, vanishes at every vertex of the simplex, maximal at `p = 0.5`. */
export function giniImpurity(proportions: readonly number[]): number {
  let sum = 0;
  for (const p of proportions) sum += p * (1 - p);
  return sum;
}

/**
 * The extracted book text for 14.32 reads `Q(T) = sum_k p_k ln p_k`, with no leading
 * minus sign. That contradicts the book's own next sentence, that this quantity
 * "vanishes for p = 0 and p = 1" and "has a maximum at p = 0.5": `sum p ln p` is <= 0
 * everywhere and is MINIMISED (most negative), not maximised, at p = 0.5 (two classes
 * at 0.5 gives `ln(0.5) = -0.693`, below the boundary value 0 at p in {0, 1}). That next
 * sentence is exactly what pins the sign: the standard Shannon-entropy form below, with
 * the leading minus sign, is the one that actually vanishes at {0, 1} and peaks at 0.5
 * as the book itself describes, so the extracted 14.32 is an extraction artefact (our
 * PDF-to-text pipeline drops superscripts and, here, a leading minus sign; the printed
 * book is fine), not an error in the book. `0 * ln(0)` is taken to be 0 by convention.
 */
export function crossEntropyImpurity(proportions: readonly number[]): number {
  let sum = 0;
  for (const p of proportions) {
    if (p > 0) sum += p * Math.log(p);
  }
  return -sum;
}

/** PRML: `1 - max_k p_k`, vanishes at every vertex of the simplex, maximal at `p = 0.5`. */
export function misclassificationImpurity(proportions: readonly number[]): number {
  return 1 - Math.max(...proportions);
}

export type ImpurityMeasure = 'gini' | 'crossEntropy';

function impurityFn(measure: ImpurityMeasure): (proportions: readonly number[]) => number {
  return measure === 'gini' ? giniImpurity : crossEntropyImpurity;
}

function classificationStat(labels: readonly number[], indices: readonly number[], numClasses: number): ClassificationStat {
  const counts = new Array(numClasses).fill(0) as number[];
  for (const i of indices) counts[labels[i]!]! += 1;
  return { counts };
}

function proportionsOf(stat: ClassificationStat, nSamples: number): number[] {
  return stat.counts.map((c) => c / nSamples);
}

/** Ties keep the lowest class index (same convention as `kmeansAssign`). */
function majorityFromCounts(counts: readonly number[]): number {
  let best = 0;
  for (let k = 1; k < counts.length; k++) {
    if (counts[k]! > counts[best]!) best = k;
  }
  return best;
}

function buildClassificationNode(
  X: Mat,
  labels: readonly number[],
  numClasses: number,
  measure: ImpurityMeasure,
  indices: readonly number[],
  depth: number,
  options: { minLeafSize: number; maxDepth: number },
): CartNode<number, ClassificationStat> {
  const nSamples = indices.length;
  const stat = classificationStat(labels, indices, numClasses);
  const leaf = (): CartNode<number, ClassificationStat> => ({
    kind: 'leaf',
    value: majorityFromCounts(stat.counts),
    nSamples,
    stat,
  });

  if (nSamples < 2 * options.minLeafSize || depth >= options.maxDepth) return leaf();

  const impurity = impurityFn(measure);
  const split = bestSplit(X, indices, options.minLeafSize, (leftIndices, rightIndices) => {
    const pLeft = proportionsOf(classificationStat(labels, leftIndices, numClasses), leftIndices.length);
    const pRight = proportionsOf(classificationStat(labels, rightIndices, numClasses), rightIndices.length);
    return (leftIndices.length / nSamples) * impurity(pLeft) + (rightIndices.length / nSamples) * impurity(pRight);
  });
  if (!split) return leaf();

  const leftIndices = indices.filter((i) => X[i]![split.featureIndex]! <= split.threshold);
  const rightIndices = indices.filter((i) => X[i]![split.featureIndex]! > split.threshold);
  return {
    kind: 'split',
    featureIndex: split.featureIndex,
    threshold: split.threshold,
    left: buildClassificationNode(X, labels, numClasses, measure, leftIndices, depth + 1, options),
    right: buildClassificationNode(X, labels, numClasses, measure, rightIndices, depth + 1, options),
    nSamples,
    stat,
  };
}

/** Labels are integers `0..numClasses-1`; leaf value is the majority class, split minimises weighted child impurity. */
export function fitClassificationTree(
  X: Mat,
  labels: readonly number[],
  numClasses: number,
  impurity: ImpurityMeasure,
  options?: CartOptions,
): CartNode<number, ClassificationStat> {
  return buildClassificationNode(X, labels, numClasses, impurity, X.map((_, i) => i), 0, resolveOptions(options));
}

export function predictClassificationTree(tree: CartNode<number, ClassificationStat>, x: Vec): number {
  let node = tree;
  while (node.kind === 'split') {
    node = x[node.featureIndex]! <= node.threshold ? node.left : node.right;
  }
  return node.value;
}

interface PruneResult<L, S> {
  readonly node: CartNode<L, S>;
  readonly cost: number;
  readonly leafCount: number;
}

interface PruneOps<L, S> {
  /** The cost of this node's own region if it were collapsed to a single leaf. */
  readonly leafCost: (stat: S, nSamples: number) => number;
  /** The value a collapsed leaf at this node would predict. */
  readonly leafValue: (stat: S, nSamples: number) => L;
}

/**
 * Weakest-link cost-complexity pruning, PRML 14.29-14.31, generic over what a leaf costs:
 * every node already carries its own region's aggregated `stat` from fitting, so this
 * never re-scans data or recombines children's statistics, only compares `ops.leafCost`
 * of keeping a subtree (its leaves' costs plus `lambda` per leaf) against collapsing it
 * to one leaf (that region's own cost plus `lambda`), collapsing on a tie (`<=`).
 */
function pruneNode<L, S>(node: CartNode<L, S>, lambda: number, ops: PruneOps<L, S>): PruneResult<L, S> {
  if (node.kind === 'leaf') {
    return { node, cost: ops.leafCost(node.stat, node.nSamples), leafCount: 1 };
  }
  const left = pruneNode(node.left, lambda, ops);
  const right = pruneNode(node.right, lambda, ops);
  const keepCost = left.cost + right.cost + lambda * (left.leafCount + right.leafCount);
  const collapsedCost = ops.leafCost(node.stat, node.nSamples);
  const collapseTotal = collapsedCost + lambda;

  // Prefer the simpler (collapsed) tree on an exact tie, per the cost-complexity rule.
  if (collapseTotal <= keepCost) {
    const leaf: CartNode<L, S> = {
      kind: 'leaf',
      value: ops.leafValue(node.stat, node.nSamples),
      nSamples: node.nSamples,
      stat: node.stat,
    };
    return { node: leaf, cost: collapsedCost, leafCount: 1 };
  }
  return {
    node: { ...node, left: left.node, right: right.node },
    cost: left.cost + right.cost,
    leafCount: left.leafCount + right.leafCount,
  };
}

/** PRML 14.29-14.31 cost-complexity (weakest-link) pruning for a regression tree at a fixed `lambda`. */
export function pruneRegressionTree(tree: CartNode<number, RegressionStat>, lambda: number): CartNode<number, RegressionStat> {
  return pruneNode(tree, lambda, {
    leafCost: (stat, nSamples) => regionSse(stat, nSamples),
    leafValue: (stat, nSamples) => stat.sumT / nSamples,
  }).node;
}

/**
 * Cost-complexity pruning for a classification tree at a fixed `lambda`, using the
 * misclassification COUNT as the leaf cost ("for subsequent pruning of the tree, the
 * misclassification rate is generally used", scaled by `nSamples` to stay commensurate
 * with a per-leaf integer penalty the same way regression's SSE is a total rather than a
 * mean). This is deliberately a different cost than the impurity `fitClassificationTree`
 * grew the tree on: Gini or cross-entropy are more sensitive to proportions than the
 * misclassification rate is, and can keep finding splits that reduce them without
 * reducing misclassification count at all, so a tree grown on Gini and pruned on
 * misclassification count can collapse a surprising amount of structure even at a
 * lambda close to zero. That is the pruning criterion doing its job, not a bug.
 */
export function pruneClassificationTree(
  tree: CartNode<number, ClassificationStat>,
  lambda: number,
): CartNode<number, ClassificationStat> {
  return pruneNode(tree, lambda, {
    leafCost: (stat, nSamples) => nSamples * misclassificationImpurity(proportionsOf(stat, nSamples)),
    leafValue: (stat) => majorityFromCounts(stat.counts),
  }).node;
}
