import {
  backpropGradientBatch,
  flattenWeights,
  forwardPass,
  gradientFromTrace,
  initializeWeights,
  mdnComponentLogPdf,
  mdnOutputGradients,
  mdnParamsFromOutput,
  norm,
  pcg32,
  propagateDeltas,
  standardNormal,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
  type NetworkWeights,
} from '@prml/math';

export const N = 150;
export const REG_SPEC: NetworkSpec = { layerSizes: [1, 10, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
export const MDN_SPEC: NetworkSpec = { layerSizes: [1, 10, 9], hiddenActivation: 'tanh', outputActivation: 'linear' };
export const K = 3;
const REG_LR = 0.3;
const REG_STEPS = 2000;
const MDN_LR = 0.05;
const MDN_STEPS = 6000;
const GRAD_CLIP = 50;

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

function trainRegression(): number[] {
  let flat = flattenWeights(initializeWeights(REG_SPEC, pcg32(20260621), 1.0));
  for (let step = 0; step < REG_STEPS; step++) {
    const weights = unflattenWeights(REG_SPEC, flat);
    const grad = flattenWeights(backpropGradientBatch(REG_SPEC, weights, DATASET, 'sumSquared'));
    flat = flat.map((w, i) => w - (REG_LR / N) * grad[i]!);
  }
  return flat;
}

/**
 * Spreads the K mean-output biases across the target range and starts variance small,
 * because a symmetric zero initialisation lets every component chase the same average
 * and never specialise onto a separate branch within a few thousand plain-gradient steps.
 */
function initMdnWeights(): number[] {
  const weights = initializeWeights(MDN_SPEC, pcg32(20260622), 1.0).map((layer) => layer.map((row) => [...row]));
  const lastLayer = weights[weights.length - 1]!;
  const meanTargets = Array.from({ length: K }, (_, j) => 0.15 + (0.7 * j) / (K - 1));
  for (let j = 0; j < K; j++) {
    lastLayer[2 * K + j]![0] = meanTargets[j]!;
    lastLayer[K + j]![0] = -1;
  }
  return flattenWeights(weights);
}

function mdnGradientBatch(weights: NetworkWeights): number[][][] {
  const zeros = weights.map((layer) => layer.map((row) => row.map(() => 0)));
  for (let n = 0; n < N; n++) {
    const input = DATASET.inputs[n]!;
    const target = DATASET.targets[n]!;
    const { activations } = forwardPass(MDN_SPEC, weights, input);
    const raw = activations[activations.length - 1]!;
    const params = mdnParamsFromOutput(raw, K, 1);
    const grads = mdnOutputGradients(params, target);
    const terminal = [...grads.dMixing, ...grads.dSigma, ...grads.dMeans.flat()];
    const deltas = propagateDeltas(MDN_SPEC, weights, activations, terminal);
    const grad = gradientFromTrace(weights, { activations, deltas });
    for (let l = 0; l < zeros.length; l++) {
      for (let j = 0; j < zeros[l]!.length; j++) {
        for (let i = 0; i < zeros[l]![j]!.length; i++) {
          zeros[l]![j]![i] = zeros[l]![j]![i]! + grad[l]![j]![i]!;
        }
      }
    }
  }
  return zeros;
}

function trainMdn(): number[] {
  let flat = initMdnWeights();
  for (let step = 0; step < MDN_STEPS; step++) {
    const weights = unflattenWeights(MDN_SPEC, flat);
    const grad = flattenWeights(mdnGradientBatch(weights));
    const gradNorm = norm(grad);
    const scale = gradNorm > GRAD_CLIP ? GRAD_CLIP / gradNorm : 1;
    flat = flat.map((w, i) => w - (MDN_LR / N) * grad[i]! * scale);
  }
  return flat;
}

export const REG_WEIGHTS = unflattenWeights(REG_SPEC, trainRegression());
export const MDN_WEIGHTS = unflattenWeights(MDN_SPEC, trainMdn());

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
