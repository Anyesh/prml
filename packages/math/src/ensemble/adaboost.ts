import { fitDecisionStump, stumpPredict, type DecisionStump } from './stumps.js';
import type { Mat, Vec } from '../types.js';

export interface AdaBoostRound {
  readonly weights: readonly number[]; // w_n^(m) used to fit this round, PRML 14.15
  readonly stump: DecisionStump;
  readonly epsilon: number; // 14.16
  readonly alpha: number; // 14.17
}

export interface AdaBoostFit {
  readonly rounds: readonly AdaBoostRound[];
}

/** ln((1-eps)/eps) diverges to +-infinity exactly at eps = 0 and eps = 1, so alpha is computed from eps clamped just inside that range. */
const EPSILON_FLOOR = 1e-10;

/** PRML 14.15-14.18, M rounds, weights initialised to 1/N (14.1). */
export function adaBoostFit(X: Mat, targets: Vec, rounds: number): AdaBoostFit {
  const n = X.length;
  let weights: number[] = new Array(n).fill(1 / n);
  const result: AdaBoostRound[] = [];

  for (let m = 0; m < rounds; m++) {
    const roundWeights = [...weights];
    const { stump, weightedError: epsilon } = fitDecisionStump(X, targets, roundWeights);
    const clamped = Math.min(Math.max(epsilon, EPSILON_FLOOR), 1 - EPSILON_FLOOR);
    const alpha = Math.log((1 - clamped) / clamped);

    result.push({ weights: roundWeights, stump, epsilon, alpha });

    // PRML 14.18: a correctly classified point's weight is unchanged (exp(0) = 1), never
    // shrunk, and weights are never renormalised because epsilon's ratio-of-sums already
    // makes the algorithm scale-invariant.
    weights = weights.map((w, i) => {
      const misclassified = stumpPredict(stump, X[i]!) !== targets[i];
      return misclassified ? w * Math.exp(alpha) : w;
    });
  }

  return { rounds: result };
}

/** PRML 14.21 with the 1/2 factor: f_m(x) = (1/2) sum_{l=1}^{upToRound} alpha_l y_l(x). Defaults upToRound to all rounds. */
export function adaBoostFunctionValue(fit: AdaBoostFit, x: Vec, upToRound?: number): number {
  const limit = upToRound ?? fit.rounds.length;
  let sum = 0;
  for (let l = 0; l < limit; l++) {
    const round = fit.rounds[l]!;
    sum += round.alpha * stumpPredict(round.stump, x);
  }
  return 0.5 * sum;
}

/** PRML 14.19: sign of the sum without the 1/2 factor, same sign as adaBoostFunctionValue since 1/2 > 0. */
export function adaBoostPredict(fit: AdaBoostFit, x: Vec, upToRound?: number): 1 | -1 {
  return adaBoostFunctionValue(fit, x, upToRound) >= 0 ? 1 : -1;
}
