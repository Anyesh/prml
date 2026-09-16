import type { Mat, Vec } from '../types.js';
import { eigSym } from '../linalg/decompose.js';
import { dot, vecSub } from '../linalg/core.js';

export type KernelPcaKernel = (a: Vec, b: Vec) => number;

/** PRML 12.86's example: `exp(-gamma ||x - x'||^2)`. */
export function kernelPcaRbfKernel(gamma: number): KernelPcaKernel {
  return (a, b) => {
    const diff = vecSub(a, b);
    return Math.exp(-gamma * dot(diff, diff));
  };
}

export function kernelPcaGramMatrix(data: Mat, kernel: KernelPcaKernel): Mat {
  const n = data.length;
  const k: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const v = kernel(data[i]!, data[j]!);
      k[i]![j] = v;
      k[j]![i] = v;
    }
  }
  return k;
}

/**
 * PRML 12.83-12.85: centres a Gram matrix as if its feature vectors had zero mean,
 * without ever forming the (possibly infinite-dimensional) feature map itself.
 */
export function centerGramMatrix(k: Mat): Mat {
  const n = k.length;
  const rowMeans = k.map((row) => row.reduce((a, b) => a + b, 0) / n);
  const grandMean = rowMeans.reduce((a, b) => a + b, 0) / n;
  return k.map((row, i) => row.map((v, j) => v - rowMeans[i]! - rowMeans[j]! + grandMean));
}

export interface KernelPcaModel {
  readonly data: Mat;
  readonly kernel: KernelPcaKernel;
  readonly rowMeans: Vec;
  readonly grandMean: number;
  /** Rows are components, normalised per PRML 12.81; only strictly positive eigenvalues are kept. */
  readonly alphas: Mat;
  readonly eigenvalues: Vec;
}

/**
 * PRML 12.80-12.81: the reduced eigenproblem `K_tilde a_i = lambda_i N a_i` on the
 * centred Gram matrix, normalised so `lambda_i N a_i^T a_i = 1`. `eigSym`'s descending
 * order and largest-magnitude-positive sign convention carry through the (positive)
 * normalising scale unchanged, so no separate sign-fixing step is needed here.
 */
export function kernelPcaFit(data: Mat, kernel: KernelPcaKernel, numComponents: number): KernelPcaModel {
  const n = data.length;
  const k = kernelPcaGramMatrix(data, kernel);
  const rowMeans = k.map((row) => row.reduce((a, b) => a + b, 0) / n);
  const grandMean = rowMeans.reduce((a, b) => a + b, 0) / n;
  const kCentered = centerGramMatrix(k);
  const { values, vectors } = eigSym(kCentered);

  const eigenvalues: number[] = [];
  const alphas: number[][] = [];
  for (let i = 0; i < values.length && eigenvalues.length < numComponents; i++) {
    const lambda = values[i]!;
    if (lambda <= 1e-8) continue;
    const v = vectors[i]!;
    const scale = 1 / Math.sqrt(lambda * n * dot(v, v));
    eigenvalues.push(lambda);
    alphas.push(v.map((x) => x * scale));
  }

  return { data, kernel, rowMeans, grandMean, alphas, eigenvalues };
}

function centerKernelVector(kx: Vec, rowMeans: Vec, grandMean: number): Vec {
  const n = kx.length;
  const kxMean = kx.reduce((a, b) => a + b, 0) / n;
  return kx.map((v, i) => v - rowMeans[i]! - kxMean + grandMean);
}

/** PRML 12.82: `y_i(x) = sum_n a_in k(x, x_n)`, on the centred kernel vector between `x` and the training set. */
export function kernelPcaProject(model: KernelPcaModel, x: Vec): Vec {
  const kx = model.data.map((xn) => model.kernel(x, xn));
  const kxCentered = centerKernelVector(kx, model.rowMeans, model.grandMean);
  return model.alphas.map((a) => dot(a, kxCentered));
}

/** `kernelPcaProject` applied to every training point, one row per point. */
export function kernelPcaTrainingProjections(model: KernelPcaModel): Mat {
  return model.data.map((x) => kernelPcaProject(model, x));
}
