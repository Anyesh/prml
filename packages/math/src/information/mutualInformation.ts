import type { Mat } from '../types.js';

function rowMarginals(joint: Mat): number[] {
  return joint.map((row) => row.reduce((s, v) => s + v, 0));
}

function columnMarginals(joint: Mat): number[] {
  const cols = joint[0]?.length ?? 0;
  const out = new Array<number>(cols).fill(0);
  for (const row of joint) {
    for (let j = 0; j < cols; j++) out[j]! += row[j]!;
  }
  return out;
}

/**
 * PRML 1.120, for a discrete joint table `joint[i][j] = p(x_i, y_j)`: the KL divergence
 * between the joint and the product of its own marginals, summed directly rather than
 * routed through `klDivergence` on a flattened table, because the zero entries of an
 * outer product of two nonzero marginals are never actually zero, so the "p_i = 0 skips"
 * guard that `klDivergence` needs never applies here.
 */
export function mutualInformation(joint: Mat): number {
  const pX = rowMarginals(joint);
  const pY = columnMarginals(joint);
  let sum = 0;
  for (let i = 0; i < joint.length; i++) {
    for (let j = 0; j < joint[i]!.length; j++) {
      const pij = joint[i]![j]!;
      if (pij > 0) sum += pij * Math.log(pij / (pX[i]! * pY[j]!));
    }
  }
  return sum;
}

/**
 * PRML 1.111-1.112, `H[Y|X] = H[X,Y] - H[X]`, evaluated directly from the joint table
 * rather than by calling `discreteEntropy` twice, so this module has no dependency on
 * `entropy.ts` and the two files can be read independently.
 */
export function conditionalEntropy(joint: Mat): number {
  const pX = rowMarginals(joint);
  let jointEntropy = 0;
  for (const row of joint) {
    for (const pij of row) {
      if (pij > 0) jointEntropy -= pij * Math.log(pij);
    }
  }
  let marginalEntropy = 0;
  for (const pi of pX) {
    if (pi > 0) marginalEntropy -= pi * Math.log(pi);
  }
  return jointEntropy - marginalEntropy;
}
