import type { Mat, Vec } from '../types.js';

export interface DecisionStump {
  readonly featureIndex: number;
  readonly threshold: number;
  readonly polarity: 1 | -1;
}

/** `s` if `x[j] > theta` else `-s`, the decision tree with a single node used as AdaBoost's weak learner. */
export function stumpPredict(stump: DecisionStump, x: Vec): 1 | -1 {
  const above = x[stump.featureIndex]! > stump.threshold;
  return (above ? stump.polarity : (-stump.polarity as 1 | -1)) as 1 | -1;
}

export interface StumpFit {
  readonly stump: DecisionStump;
  /** PRML 14.16: weighted misclassification rate of the returned stump. */
  readonly weightedError: number;
}

/**
 * Exhaustive search minimizing PRML 14.15/14.16, weights need not sum to 1. Candidates
 * are enumerated feature index ascending, then threshold ascending (midpoints between
 * consecutive distinct sorted values of that feature), then polarity +1 before -1, and a
 * candidate only replaces the running best on a strict improvement, so an exact tie
 * resolves to whichever candidate came first in that order. AdaBoost relies on this
 * being deterministic across identical inputs.
 */
export function fitDecisionStump(X: Mat, targets: Vec, weights: Vec): StumpFit {
  const n = X.length;
  const dimension = X[0]?.length ?? 0;
  let totalWeight = 0;
  for (const w of weights) totalWeight += w;

  let best: StumpFit | undefined;

  for (let j = 0; j < dimension; j++) {
    const values = Array.from(new Set(X.map((row) => row[j]!))).sort((a, b) => a - b);
    if (values.length < 2) continue;

    for (let i = 1; i < values.length; i++) {
      const theta = (values[i - 1]! + values[i]!) / 2;

      for (const polarity of [1, -1] as const) {
        let misweight = 0;
        for (let m = 0; m < n; m++) {
          const raw = X[m]![j]! > theta ? 1 : -1;
          const predicted = polarity * raw;
          if (predicted !== targets[m]) misweight += weights[m]!;
        }
        const weightedError = misweight / totalWeight;
        if (!best || weightedError < best.weightedError) {
          best = { stump: { featureIndex: j, threshold: theta, polarity }, weightedError };
        }
      }
    }
  }

  if (!best) {
    throw new Error('fitDecisionStump: every feature has fewer than two distinct values');
  }
  return best;
}
