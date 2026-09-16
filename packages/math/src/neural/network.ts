import type { Rng, Vec } from '../types.js';
import { standardNormal } from '../rng.js';
import { applyHidden, applyOutput } from './activations.js';
import type { ForwardTrace, NetworkSpec, NetworkWeights } from './types.js';

/** Prepends the bias unit `z_0 = 1`, PRML's convention (eq 5.8) for absorbing `w_{j0}`. */
function augment(z: Vec): number[] {
  return [1, ...z];
}

/** Total scalar weights across every layer, `Σ_l layerSizes[l+1] * (layerSizes[l] + 1)`. */
export function countWeights(spec: NetworkSpec): number {
  let total = 0;
  for (let l = 0; l < spec.layerSizes.length - 1; l++) {
    total += spec.layerSizes[l + 1]! * (spec.layerSizes[l]! + 1);
  }
  return total;
}

/**
 * Small random weights, PRML's own recommendation (p.268, the initialisation step
 * every training run in this chapter starts from): scaled by the reciprocal square
 * root of the fan-in, so a layer with many inputs does not start with a much larger
 * activation than one with few.
 */
export function initializeWeights(spec: NetworkSpec, rng: Rng, scale = 1): NetworkWeights {
  const layers: number[][][] = [];
  for (let l = 0; l < spec.layerSizes.length - 1; l++) {
    const nIn = spec.layerSizes[l]!;
    const nOut = spec.layerSizes[l + 1]!;
    const fanScale = scale / Math.sqrt(nIn + 1);
    const layer: number[][] = [];
    for (let j = 0; j < nOut; j++) {
      const row: number[] = [];
      for (let i = 0; i < nIn + 1; i++) row.push(standardNormal(rng) * fanScale);
      layer.push(row);
    }
    layers.push(layer);
  }
  return layers;
}

export function flattenWeights(weights: NetworkWeights): number[] {
  const out: number[] = [];
  for (const layer of weights) for (const row of layer) for (const w of row) out.push(w);
  return out;
}

/** Inverse of `flattenWeights`, reading back in the same layer-then-row-then-column order. */
export function unflattenWeights(spec: NetworkSpec, flat: Vec): NetworkWeights {
  const layers: number[][][] = [];
  let cursor = 0;
  for (let l = 0; l < spec.layerSizes.length - 1; l++) {
    const nIn = spec.layerSizes[l]!;
    const nOut = spec.layerSizes[l + 1]!;
    const layer: number[][] = [];
    for (let j = 0; j < nOut; j++) {
      const row = flat.slice(cursor, cursor + nIn + 1);
      cursor += nIn + 1;
      layer.push(row as number[]);
    }
    layers.push(layer);
  }
  return layers;
}

/**
 * Forward propagation, PRML 5.48-5.49 layer by layer, with the final layer's
 * activation switched to `spec.outputActivation` instead of the hidden one. Every
 * intermediate `z` and `a` is kept (not just the output) because `backpropGradient`
 * and the exact-Hessian R-operator both need them.
 */
export function forwardPass(spec: NetworkSpec, weights: NetworkWeights, input: Vec): ForwardTrace {
  const activations: Vec[] = [input];
  const preActivations: Vec[] = [[]];
  const numLayers = weights.length;

  for (let l = 0; l < numLayers; l++) {
    const augmented = augment(activations[l]!);
    const a = weights[l]!.map((row) => dotRow(row, augmented));
    const isOutput = l === numLayers - 1;
    const z = isOutput ? applyOutput(spec.outputActivation, a) : applyHidden(spec.hiddenActivation, a);
    preActivations.push(a);
    activations.push(z);
  }

  return { activations, preActivations, output: activations[numLayers]! };
}

function dotRow(row: Vec, x: Vec): number {
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += row[i]! * x[i]!;
  return sum;
}

export function predict(spec: NetworkSpec, weights: NetworkWeights, input: Vec): Vec {
  return forwardPass(spec, weights, input).output;
}

export { augment };
