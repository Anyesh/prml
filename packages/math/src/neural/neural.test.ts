import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import type { Mat, Vec } from '../types.js';
import { forwardPass } from './network.js';
import {
  crossEntropyBinarySingleFromActivations,
  crossEntropySoftmaxSingle,
  networkError,
  sumSquaredErrorSingle,
} from './errors.js';
import { backpropGradientBatch, backpropGradientSingle } from './backprop.js';
import { numericalGradient } from './numericalGradient.js';
import { weightDecayGradient, weightDecayPenalty } from './regularization.js';
import { exactHessian, outerProductHessian } from './hessian.js';
import {
  mdnErrorSingle,
  mdnOutputGradients,
  mdnParamsFromOutput,
  mdnPredictiveMean,
  mdnPredictiveVariance,
  mdnResponsibilities,
} from './mdn.js';
import type { Dataset, ErrorKind, NetworkSpec, NetworkWeights } from './types.js';

interface Fixture {
  readonly cases: Record<string, unknown>[];
}

const fixture = loadFixture<Fixture>('neural');
const pick = (fn: string) => fixture.cases.filter((c) => c['fn'] === fn);

const TOL = 1e-9;

function close(actual: number, expected: number, tol = TOL): boolean {
  return Math.abs(actual - expected) <= tol * Math.max(1, Math.abs(expected));
}

function expectVecClose(actual: readonly number[], expected: readonly number[], tol = TOL) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((e, i) => expect(close(actual[i]!, e, tol)).toBe(true));
}

function expectMatClose(actual: Mat, expected: readonly (readonly number[])[], tol = TOL) {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((row, i) => expectVecClose(actual[i]!, row, tol));
}

function specOf(c: Record<string, unknown>): NetworkSpec {
  return c['spec'] as NetworkSpec;
}

function weightsOf(c: Record<string, unknown>): NetworkWeights {
  return c['weights'] as NetworkWeights;
}

describe('forwardPass', () => {
  it('matches an independent numpy forward pass for linear, logistic and softmax outputs', () => {
    for (const c of pick('forwardPass')) {
      const trace = forwardPass(specOf(c), weightsOf(c), c['input'] as Vec);
      const expected = c['expected'] as Record<string, unknown>;
      const expectedActivations = expected['activations'] as number[][];
      const expectedPre = expected['preActivations'] as number[][];
      trace.activations.forEach((a, l) => expectVecClose(a, expectedActivations[l]!));
      trace.preActivations.slice(1).forEach((a, l) => expectVecClose(a, expectedPre[l + 1]!));
      expectVecClose(trace.output, expected['output'] as number[]);
    }
  });
});

describe('error functions', () => {
  it('sum-of-squares matches PRML 5.11', () => {
    const c = pick('sumSquaredErrorSingle')[0]!;
    const got = sumSquaredErrorSingle(c['predicted'] as Vec, c['target'] as Vec);
    expect(close(got, c['expected'] as number)).toBe(true);
  });

  it('binary cross-entropy from activations matches PRML 5.21 via the softplus identity', () => {
    const c = pick('crossEntropyBinarySingleFromActivations')[0]!;
    const got = crossEntropyBinarySingleFromActivations(c['preActivations'] as Vec, c['target'] as Vec);
    expect(close(got, c['expected'] as number)).toBe(true);
  });

  it('softmax cross-entropy matches PRML 5.24', () => {
    const c = pick('crossEntropySoftmaxSingle')[0]!;
    const got = crossEntropySoftmaxSingle(c['predicted'] as Vec, c['target'] as Vec);
    expect(close(got, c['expected'] as number)).toBe(true);
  });

  it('networkError sums per-pattern error across a dataset for every canonical pairing', () => {
    for (const c of pick('networkError')) {
      const dataset: Dataset = { inputs: c['inputs'] as Mat, targets: c['targets'] as Mat };
      const got = networkError(specOf(c), weightsOf(c), dataset, c['kind'] as ErrorKind);
      expect(close(got, c['expected'] as number)).toBe(true);
    }
  });
});

describe('backpropagation', () => {
  it('matches a central difference of the error function for every canonical pairing', () => {
    for (const c of pick('backpropGradientSingle')) {
      const grad = backpropGradientSingle(specOf(c), weightsOf(c), c['input'] as Vec, c['target'] as Vec, c['kind'] as ErrorKind);
      const flat = grad.flatMap((layer) => layer.flatMap((row) => row));
      expectVecClose(flat, c['expected'] as number[]);
    }
  });

  it('batches to the central difference of the summed dataset error', () => {
    for (const c of pick('backpropGradientBatch')) {
      const dataset: Dataset = { inputs: c['inputs'] as Mat, targets: c['targets'] as Mat };
      const grad = backpropGradientBatch(specOf(c), weightsOf(c), dataset, c['kind'] as ErrorKind);
      const flat = grad.flatMap((layer) => layer.flatMap((row) => row));
      expectVecClose(flat, c['expected'] as number[]);
    }
  });
});

describe('numericalGradient', () => {
  it('matches an independent numpy central difference on a plain smooth function', () => {
    const c = pick('numericalGradient')[0]!;
    const x0 = c['x0'] as Vec;
    const epsScale = c['epsScale'] as number;
    const f = (x: Vec) => Math.sin(x[0]!) + x[0]! * x[1]! * x[1]! + Math.exp(x[2]!) + x[1]! * x[3]!;
    const got = numericalGradient(f, x0, epsScale);
    expectVecClose(got, c['expected'] as number[]);
  });
});

describe('weight decay', () => {
  it('matches PRML 5.112', () => {
    const c = pick('weightDecayPenalty')[0]!;
    const got = weightDecayPenalty(c['flatWeights'] as Vec, c['lambda'] as number);
    expect(close(got, c['expected'] as number)).toBe(true);
  });

  it('gradient is λw', () => {
    const c = pick('weightDecayGradient')[0]!;
    const got = weightDecayGradient(c['flatWeights'] as Vec, c['lambda'] as number);
    expectVecClose(got, c['expected'] as number[]);
  });
});

describe('outer-product (Levenberg-Marquardt) Hessian', () => {
  it('matches an independent numpy computation for sum-of-squares and cross-entropy, PRML 5.84-5.85', () => {
    for (const c of pick('outerProductHessian')) {
      const dataset: Dataset = { inputs: c['inputs'] as Mat, targets: [] };
      const got = outerProductHessian(specOf(c), weightsOf(c), dataset, c['kind'] as ErrorKind);
      expectMatClose(got, c['expected'] as number[][]);
    }
  });
});

describe('exact Hessian via the R-operator', () => {
  it('matches a central difference of an independent numpy analytic gradient, PRML 5.4.5/5.4.6', () => {
    for (const c of pick('exactHessian')) {
      const dataset: Dataset = { inputs: c['inputs'] as Mat, targets: c['targets'] as Mat };
      const got = exactHessian(specOf(c), weightsOf(c), dataset, c['kind'] as ErrorKind);
      expectMatClose(got, c['expected'] as number[][]);
    }
  });
});

describe('mixture density network', () => {
  it('splits raw output into mixing/sigma/means per PRML 5.150-5.152', () => {
    for (const c of pick('mdnParamsFromOutput')) {
      const params = mdnParamsFromOutput(c['raw'] as Vec, c['numComponents'] as number, c['targetDim'] as number);
      const expected = c['expected'] as Record<string, unknown>;
      expectVecClose(params.mixing, expected['mixing'] as number[]);
      expectVecClose(params.sigma, expected['sigma'] as number[]);
      expectMatClose(params.means, expected['means'] as number[][]);
    }
  });

  it('error function matches PRML 5.153', () => {
    for (const c of pick('mdnErrorSingle')) {
      const params = mdnParamsFromOutput(c['raw'] as Vec, c['numComponents'] as number, c['targetDim'] as number);
      const got = mdnErrorSingle(params, c['target'] as Vec);
      expect(close(got, c['expected'] as number)).toBe(true);
    }
  });

  it('responsibilities match PRML 5.154', () => {
    const c = pick('mdnResponsibilities')[0]!;
    const params = mdnParamsFromOutput(c['raw'] as Vec, c['numComponents'] as number, c['targetDim'] as number);
    const got = mdnResponsibilities(params, c['target'] as Vec);
    expectVecClose(got, c['expected'] as number[]);
  });

  it('output-layer gradients match a central difference of the error function, PRML 5.155-5.157', () => {
    const c = pick('mdnOutputGradients')[0]!;
    const params = mdnParamsFromOutput(c['raw'] as Vec, c['numComponents'] as number, c['targetDim'] as number);
    const got = mdnOutputGradients(params, c['target'] as Vec);
    const expected = c['expected'] as Record<string, unknown>;
    expectVecClose(got.dMixing, expected['dMixing'] as number[]);
    expectVecClose(got.dSigma, expected['dSigma'] as number[]);
    expectMatClose(got.dMeans, expected['dMeans'] as number[][]);
  });

  it('predictive mean matches PRML 5.158', () => {
    for (const c of pick('mdnPredictiveMean')) {
      const params = mdnParamsFromOutput(c['raw'] as Vec, c['numComponents'] as number, c['targetDim'] as number);
      const got = mdnPredictiveMean(params);
      expectVecClose(got, c['expected'] as number[]);
    }
  });

  it('predictive variance matches PRML 5.159-5.160', () => {
    for (const c of pick('mdnPredictiveVariance')) {
      const params = mdnParamsFromOutput(c['raw'] as Vec, c['numComponents'] as number, c['targetDim'] as number);
      const got = mdnPredictiveVariance(params);
      expect(close(got, c['expected'] as number)).toBe(true);
    }
  });
});
