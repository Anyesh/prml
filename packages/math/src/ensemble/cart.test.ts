import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import {
  type CartNode,
  type ClassificationStat,
  type RegressionStat,
  crossEntropyImpurity,
  fitClassificationTree,
  fitRegressionTree,
  giniImpurity,
  misclassificationImpurity,
  predictClassificationTree,
  predictRegressionTree,
  pruneClassificationTree,
  pruneRegressionTree,
} from './cart.js';

interface ExpectedRegressionNode {
  readonly kind: 'leaf' | 'split';
  readonly value?: number;
  readonly featureIndex?: number;
  readonly threshold?: number;
  readonly nSamples: number;
  readonly stat: RegressionStat;
  readonly left?: ExpectedRegressionNode;
  readonly right?: ExpectedRegressionNode;
}

interface ExpectedClassificationNode {
  readonly kind: 'leaf' | 'split';
  readonly value?: number;
  readonly featureIndex?: number;
  readonly threshold?: number;
  readonly nSamples: number;
  readonly stat: ClassificationStat;
  readonly left?: ExpectedClassificationNode;
  readonly right?: ExpectedClassificationNode;
}

interface RegressionSplitCase {
  readonly fn: 'regressionSplit';
  readonly name: string;
  readonly X: Mat;
  readonly t: Vec;
  readonly options: { readonly minLeafSize: number; readonly maxDepth: number };
  readonly expectedTree: ExpectedRegressionNode;
  readonly predictions: readonly { readonly x: Vec; readonly expected: number }[];
}

interface ClassificationSplitCase {
  readonly fn: 'classificationSplit';
  readonly name: string;
  readonly X: Mat;
  readonly labels: readonly number[];
  readonly numClasses: number;
  readonly impurity: 'gini' | 'crossEntropy';
  readonly options: { readonly minLeafSize: number; readonly maxDepth: number };
  readonly expectedTree: ExpectedClassificationNode;
  readonly predictions: readonly { readonly x: Vec; readonly expected: number }[];
}

interface RegressionPruneCase {
  readonly fn: 'regressionPrune';
  readonly X: Mat;
  readonly t: Vec;
  readonly options: { readonly minLeafSize: number; readonly maxDepth: number };
  readonly baseTree: ExpectedRegressionNode;
  readonly pruneCases: readonly {
    readonly lambda: number;
    readonly expectedTree: ExpectedRegressionNode;
    readonly expectedLeafCount: number;
  }[];
}

interface ClassificationPruneCase {
  readonly fn: 'classificationPrune';
  readonly X: Mat;
  readonly labels: readonly number[];
  readonly numClasses: number;
  readonly options: { readonly minLeafSize: number; readonly maxDepth: number };
  readonly baseTree: ExpectedClassificationNode;
  readonly pruneCases: readonly {
    readonly lambda: number;
    readonly expectedTree: ExpectedClassificationNode;
    readonly expectedLeafCount: number;
  }[];
}

interface ImpurityCase<Fn extends string> {
  readonly fn: Fn;
  readonly proportions: readonly number[];
  readonly expected: number;
}

type FixtureCase =
  | RegressionSplitCase
  | ClassificationSplitCase
  | RegressionPruneCase
  | ClassificationPruneCase
  | ImpurityCase<'giniImpurity'>
  | ImpurityCase<'crossEntropyImpurity'>
  | ImpurityCase<'misclassificationImpurity'>;

interface Fixture {
  readonly cases: readonly FixtureCase[];
}

const fixture = loadFixture<Fixture>('cart');
const TOL = 1e-9;

function closeTo(actual: number, expected: number, tol = TOL): void {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));
}

function expectRegressionTreeMatches(actual: CartNode<number, RegressionStat>, expected: ExpectedRegressionNode): void {
  expect(actual.kind).toBe(expected.kind);
  expect(actual.nSamples).toBe(expected.nSamples);
  closeTo(actual.stat.sumT, expected.stat.sumT);
  closeTo(actual.stat.sumTSq, expected.stat.sumTSq);
  if (actual.kind === 'leaf' && expected.kind === 'leaf') {
    closeTo(actual.value, expected.value!);
  } else if (actual.kind === 'split' && expected.kind === 'split') {
    expect(actual.featureIndex).toBe(expected.featureIndex);
    expect(actual.threshold).toBe(expected.threshold);
    expectRegressionTreeMatches(actual.left, expected.left!);
    expectRegressionTreeMatches(actual.right, expected.right!);
  } else {
    throw new Error(`tree shape mismatch: actual ${actual.kind}, expected ${expected.kind}`);
  }
}

function expectClassificationTreeMatches(
  actual: CartNode<number, ClassificationStat>,
  expected: ExpectedClassificationNode,
): void {
  expect(actual.kind).toBe(expected.kind);
  expect(actual.nSamples).toBe(expected.nSamples);
  expect(actual.stat.counts).toEqual(expected.stat.counts);
  if (actual.kind === 'leaf' && expected.kind === 'leaf') {
    expect(actual.value).toBe(expected.value);
  } else if (actual.kind === 'split' && expected.kind === 'split') {
    expect(actual.featureIndex).toBe(expected.featureIndex);
    expect(actual.threshold).toBe(expected.threshold);
    expectClassificationTreeMatches(actual.left, expected.left!);
    expectClassificationTreeMatches(actual.right, expected.right!);
  } else {
    throw new Error(`tree shape mismatch: actual ${actual.kind}, expected ${expected.kind}`);
  }
}

function pick<T extends FixtureCase['fn']>(fn: T): Extract<FixtureCase, { fn: T }>[] {
  return fixture.cases.filter((c): c is Extract<FixtureCase, { fn: T }> => c.fn === fn);
}

describe('impurity measures', () => {
  it('matches numpy for giniImpurity', () => {
    for (const c of pick('giniImpurity')) {
      closeTo(giniImpurity(c.proportions), c.expected);
    }
  });

  it('matches numpy for crossEntropyImpurity (with the sign correction to 14.32)', () => {
    for (const c of pick('crossEntropyImpurity')) {
      closeTo(crossEntropyImpurity(c.proportions), c.expected);
    }
  });

  it('matches numpy for misclassificationImpurity', () => {
    for (const c of pick('misclassificationImpurity')) {
      closeTo(misclassificationImpurity(c.proportions), c.expected);
    }
  });
});

describe('fitRegressionTree', () => {
  it('finds the unambiguous root split and matches the full reference tree and predictions', () => {
    const c = pick('regressionSplit').find((x) => x.name === 'unambiguous')!;
    const tree = fitRegressionTree(c.X, c.t, c.options);
    expectRegressionTreeMatches(tree, c.expectedTree);
    for (const p of c.predictions) {
      closeTo(predictRegressionTree(tree, p.x), p.expected);
    }
  });

  it('breaks an exact tie between two identical feature columns by picking the lower feature index', () => {
    const c = pick('regressionSplit').find((x) => x.name === 'tieBreak')!;
    const tree = fitRegressionTree(c.X, c.t, c.options);
    expectRegressionTreeMatches(tree, c.expectedTree);
    expect(tree.kind).toBe('split');
    if (tree.kind === 'split') {
      expect(tree.featureIndex).toBe(0);
    }
  });
});

describe('fitClassificationTree', () => {
  it('matches the reference tree and predictions under gini and cross-entropy impurity, with class-count stats rather than borrowed regression stats', () => {
    for (const c of pick('classificationSplit')) {
      const tree = fitClassificationTree(c.X, c.labels, c.numClasses, c.impurity, c.options);
      expectClassificationTreeMatches(tree, c.expectedTree);
      for (const p of c.predictions) {
        expect(predictClassificationTree(tree, p.x)).toBe(p.expected);
      }
    }
  });
});

describe('pruneRegressionTree', () => {
  const c = pick('regressionPrune')[0]!;
  const tree = fitRegressionTree(c.X, c.t, c.options);

  it('matches the unpruned reference tree before any pruning is applied', () => {
    expectRegressionTreeMatches(tree, c.baseTree);
  });

  it('collapses nodes under weakest-link cost-complexity pruning exactly as the reference recursion does', () => {
    for (const pc of c.pruneCases) {
      const pruned = pruneRegressionTree(tree, pc.lambda);
      expectRegressionTreeMatches(pruned, pc.expectedTree);
    }
  });

  it('prunes nothing at a near-zero lambda, collapses one subtree at a middle lambda, and collapses to the root at a large lambda', () => {
    const [none, partial, full] = c.pruneCases;
    expect(none!.expectedLeafCount).toBe(4);
    expect(partial!.expectedLeafCount).toBe(3);
    expect(full!.expectedLeafCount).toBe(1);

    const partialPruned = pruneRegressionTree(tree, partial!.lambda);
    expect(partialPruned.kind).toBe('split');
    if (partialPruned.kind === 'split') {
      expect(partialPruned.left.kind).toBe('leaf');
      expect(partialPruned.right.kind).toBe('split');
    }

    const fullyPruned = pruneRegressionTree(tree, full!.lambda);
    expect(fullyPruned.kind).toBe('leaf');
  });
});

describe('pruneClassificationTree', () => {
  const c = pick('classificationPrune')[0]!;
  const tree = fitClassificationTree(c.X, c.labels, c.numClasses, 'gini', c.options);

  it('matches the unpruned reference tree before any pruning is applied', () => {
    expectClassificationTreeMatches(tree, c.baseTree);
  });

  it('collapses nodes under misclassification-count weakest-link pruning exactly as the reference recursion does', () => {
    for (const pc of c.pruneCases) {
      const pruned = pruneClassificationTree(tree, pc.lambda);
      expectClassificationTreeMatches(pruned, pc.expectedTree);
    }
  });

  it('collapses a Gini-overgrown tree back to the true region count already at a near-zero lambda, because those extra splits reduce Gini without reducing misclassification count', () => {
    const [nearZero, stillFour, full] = c.pruneCases;
    expect(nearZero!.expectedLeafCount).toBe(4);
    expect(stillFour!.expectedLeafCount).toBe(4);
    expect(full!.expectedLeafCount).toBe(1);

    const prunedNearZero = pruneClassificationTree(tree, nearZero!.lambda);
    let leaves = 0;
    (function count(node: CartNode<number, ClassificationStat>) {
      if (node.kind === 'leaf') leaves += 1;
      else {
        count(node.left);
        count(node.right);
      }
    })(prunedNearZero);
    expect(leaves).toBe(4);
    expect(leaves).toBeLessThan(countLeaves(tree));
  });
});

function countLeaves(node: CartNode<number, ClassificationStat>): number {
  return node.kind === 'leaf' ? 1 : countLeaves(node.left) + countLeaves(node.right);
}
