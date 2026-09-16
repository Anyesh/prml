import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  mixtureExpertsFitEM,
  mixtureExpertsLogLikelihood,
  mixtureExpertsMStep,
  mixtureExpertsResponsibilities,
  type ExpertsParams,
} from './mixtureExperts.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('mixtureExperts');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

const TOL = 1e-9;
function assertClose(actual: number, expected: number, tol = TOL) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));
}

function assertParamsClose(actual: ExpertsParams, expected: ExpertsParams, tol = TOL) {
  expect(actual.expertWeights).toHaveLength(expected.expertWeights.length);
  actual.expertWeights.forEach((row, k) => row.forEach((v, d) => assertClose(v, expected.expertWeights[k]![d]!, tol)));
  assertClose(actual.beta, expected.beta, tol);
  expect(actual.gateWeights).toHaveLength(expected.gateWeights.length);
  actual.gateWeights.forEach((row, k) => row.forEach((v, d) => assertClose(v, expected.gateWeights[k]![d]!, tol)));
}

describe('mixtureExpertsResponsibilities', () => {
  it('matches an independent softmax of the gated Gaussian log terms (PRML 14.53, K=3)', () => {
    const c = casesFor('mixtureExpertsResponsibilities')[0]!;
    const designExperts = c['designExperts'] as number[][];
    const designGate = c['designGate'] as number[][];
    const targets = c['targets'] as number[];
    const params = c['params'] as ExpertsParams;
    const expected = c['expected'] as number[][];

    const actual = mixtureExpertsResponsibilities(designExperts, designGate, targets, params);
    expect(actual).toHaveLength(expected.length);
    actual.forEach((row, n) => {
      row.forEach((v, k) => assertClose(v, expected[n]![k]!));
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    });
  });
});

describe('mixtureExpertsLogLikelihood', () => {
  it('matches an independent scipy norm.logpdf / expit reference via logSumExp', () => {
    const c = casesFor('mixtureExpertsLogLikelihood')[0]!;
    const designExperts = c['designExperts'] as number[][];
    const designGate = c['designGate'] as number[][];
    const targets = c['targets'] as number[];
    const params = c['params'] as ExpertsParams;
    assertClose(mixtureExpertsLogLikelihood(designExperts, designGate, targets, params), c['expected'] as number);
  });
});

describe('mixtureExpertsMStep', () => {
  it('matches independent weighted least squares experts, pooled beta, and a Newton-solved gate', () => {
    const c = casesFor('mixtureExpertsMStep')[0]!;
    const designExperts = c['designExperts'] as number[][];
    const designGate = c['designGate'] as number[][];
    const targets = c['targets'] as number[];
    const responsibilities = c['responsibilities'] as number[][];
    const expected = c['expected'] as ExpertsParams;

    const actual = mixtureExpertsMStep(designExperts, designGate, targets, responsibilities);
    assertParamsClose(actual, expected);
  });
});

describe('mixtureExpertsFitEM', () => {
  it('reproduces a hand-rolled E-step-then-M-step iterate trace', () => {
    const c = casesFor('mixtureExpertsEmTrace')[0]!;
    const designExperts = c['designExperts'] as number[][];
    const designGate = c['designGate'] as number[][];
    const targets = c['targets'] as number[];
    const initialParams = c['initialParams'] as ExpertsParams;
    const steps = c['steps'] as number;
    const expectedTrace = c['expectedTrace'] as Array<{
      responsibilities: number[][];
      logLikelihoodBeforeMStep: number;
      paramsAfterMStep: ExpertsParams;
    }>;

    const fit = mixtureExpertsFitEM(designExperts, designGate, targets, initialParams, steps);

    expectedTrace.forEach((expected, i) => {
      fit.responsibilitiesHistory[i]!.forEach((row, n) =>
        row.forEach((v, k) => assertClose(v, expected.responsibilities[n]![k]!)),
      );
      assertClose(fit.logLikelihoodHistory[i]!, expected.logLikelihoodBeforeMStep);
      assertParamsClose(fit.paramsHistory[i + 1]!, expected.paramsAfterMStep);
    });
  });

  it('never decreases the log-likelihood from one iterate to the next', () => {
    const c = casesFor('mixtureExpertsEmTrace')[0]!;
    const designExperts = c['designExperts'] as number[][];
    const designGate = c['designGate'] as number[][];
    const targets = c['targets'] as number[];
    const initialParams = c['initialParams'] as ExpertsParams;
    const steps = c['steps'] as number;

    const fit = mixtureExpertsFitEM(designExperts, designGate, targets, initialParams, steps);
    for (let i = 1; i < fit.logLikelihoodHistory.length; i++) {
      expect(fit.logLikelihoodHistory[i]!).toBeGreaterThanOrEqual(fit.logLikelihoodHistory[i - 1]! - 1e-9);
    }
  });
});
