import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  mixtureLinearRegressionFitEM,
  mixtureLinearRegressionLogLikelihood,
  mixtureLinearRegressionMStep,
  mixtureLinearRegressionResponsibilities,
  type LinearMixtureParams,
} from './mixtureLinearRegression.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('mixtureLinearRegression');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

const TOL = 1e-9;
function closeTo(actual: number, expected: number) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(TOL * Math.max(1, Math.abs(expected)));
}

function expectParamsClose(actual: LinearMixtureParams, expected: LinearMixtureParams) {
  actual.weights.forEach((row, k) => row.forEach((v, d) => closeTo(v, expected.weights[k]![d]!)));
  actual.mixing.forEach((v, k) => closeTo(v, expected.mixing[k]!));
  closeTo(actual.beta, expected.beta);
}

describe('mixtureLinearRegressionResponsibilities', () => {
  it('matches PRML 14.37 at a fixed, unconverged mixture', () => {
    const c = casesFor('responsibilities')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const params = c['params'] as LinearMixtureParams;
    const expected = c['expected'] as number[][];
    const actual = mixtureLinearRegressionResponsibilities(design, targets, params);
    actual.forEach((row, n) => row.forEach((v, k) => closeTo(v, expected[n]![k]!)));
  });
});

describe('mixtureLinearRegressionLogLikelihood', () => {
  it('matches PRML 14.35 at the same fixed mixture', () => {
    const c = casesFor('logLikelihood')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const params = c['params'] as LinearMixtureParams;
    closeTo(mixtureLinearRegressionLogLikelihood(design, targets, params), c['expected'] as number);
  });
});

describe('mixtureLinearRegressionMStep', () => {
  it('matches PRML 14.38 (mixing), 14.42 (weights) and 14.44 (beta, from the updated weights)', () => {
    const c = casesFor('mStep')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const responsibilities = c['responsibilities'] as number[][];
    const expected = c['expected'] as LinearMixtureParams;
    const actual = mixtureLinearRegressionMStep(design, targets, responsibilities);
    expectParamsClose(actual, expected);
  });
});

describe('mixtureLinearRegressionFitEM', () => {
  it('reproduces a hand-rolled E-step-then-M-step iterate trace', () => {
    const c = casesFor('fitEM')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const initialParams = c['initialParams'] as LinearMixtureParams;
    const steps = c['steps'] as number;
    const expectedTrace = c['expectedTrace'] as Array<{
      responsibilities: number[][];
      logLikelihoodBeforeMStep: number;
      paramsAfterMStep: LinearMixtureParams;
    }>;

    const fit = mixtureLinearRegressionFitEM(design, targets, initialParams, steps);

    expect(fit.paramsHistory).toHaveLength(steps + 1);
    expect(fit.responsibilitiesHistory).toHaveLength(steps);
    expect(fit.logLikelihoodHistory).toHaveLength(steps);
    expectParamsClose(fit.paramsHistory[0]!, initialParams);

    expectedTrace.forEach((expected, i) => {
      fit.responsibilitiesHistory[i]!.forEach((row, n) =>
        row.forEach((v, k) => closeTo(v, expected.responsibilities[n]![k]!)),
      );
      closeTo(fit.logLikelihoodHistory[i]!, expected.logLikelihoodBeforeMStep);
      expectParamsClose(fit.paramsHistory[i + 1]!, expected.paramsAfterMStep);
    });
  });

  it('never decreases the log-likelihood from one iterate to the next', () => {
    const c = casesFor('fitEM')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const initialParams = c['initialParams'] as LinearMixtureParams;
    const fit = mixtureLinearRegressionFitEM(design, targets, initialParams, 4);
    let previous = -Infinity;
    for (const ll of fit.logLikelihoodHistory) {
      expect(ll).toBeGreaterThanOrEqual(previous - 1e-9);
      previous = ll;
    }
  });
});
