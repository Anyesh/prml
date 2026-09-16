import {
  forwardPass,
  mdnComponentLogPdf,
  mdnParamsFromOutput,
  pcg32,
  standardNormal,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
} from '@prml/math';

export const N = 150;
export const REG_SPEC: NetworkSpec = { layerSizes: [1, 10, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
export const MDN_SPEC: NetworkSpec = { layerSizes: [1, 10, 9], hiddenActivation: 'tanh', outputActivation: 'linear' };
export const K = 3;

function buildInverseDataset(): Dataset {
  const rng = pcg32(20260620);
  const inputs: number[][] = [];
  const targets: number[][] = [];
  for (let i = 0; i < N; i++) {
    const x = rng.next();
    const t = x + 0.3 * Math.sin(2 * Math.PI * x) + 0.03 * standardNormal(rng);
    inputs.push([t]);
    targets.push([x]);
  }
  return { inputs, targets };
}

export const DATASET = buildInverseDataset();

import trainedWeights from './mdnToyWeights.json' with { type: 'json' };

/**
 * Training is deterministic from a fixed seed, so the fitted weights are constants and are
 * committed rather than recomputed. Running the six thousand gradient steps at module scope
 * cost about six seconds of blocking work in the reader's browser on every hydration, for an
 * answer that never changes. `mdnTraining.test.ts` re-runs the training and fails if this
 * file drifts from it.
 */
export const REG_WEIGHTS = unflattenWeights(REG_SPEC, trainedWeights.regression);
export const MDN_WEIGHTS = unflattenWeights(MDN_SPEC, trainedWeights.mdn);

export function mdnParamsAt(t: number) {
  const raw = forwardPass(MDN_SPEC, MDN_WEIGHTS, [t]).output;
  return mdnParamsFromOutput(raw, K, 1);
}

export function regressionAt(t: number): number {
  return forwardPass(REG_SPEC, REG_WEIGHTS, [t]).output[0]!;
}

export function mdnComponentDensity(t: number, x: number): number {
  const params = mdnParamsAt(t);
  let density = 0;
  for (let k = 0; k < K; k++) {
    density += params.mixing[k]! * Math.exp(mdnComponentLogPdf([x], params.means[k]!, params.sigma[k]!));
  }
  return density;
}
