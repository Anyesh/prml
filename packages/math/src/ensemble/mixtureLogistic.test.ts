import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  mixtureLogisticFitEM,
  mixtureLogisticLogLikelihood,
  mixtureLogisticMStep,
  mixtureLogisticResponsibilities,
  type LogisticMixtureParams,
} from './mixtureLogistic.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('mixtureLogistic');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

const TOL = 1e-9;
function assertClose(actual: number, expected: number, tol = TOL) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(expected)));
}

function assertParamsClose(actual: LogisticMixtureParams, expected: LogisticMixtureParams, tol = TOL) {
  expect(actual.weights).toHaveLength(expected.weights.length);
  actual.weights.forEach((row, k) => row.forEach((v, d) => assertClose(v, expected.weights[k]![d]!, tol)));
  actual.mixing.forEach((v, k) => assertClose(v, expected.mixing[k]!, tol));
}

describe('mixtureLogisticResponsibilities', () => {
  it('matches an independent softmax of the per-component log terms (PRML 14.48)', () => {
    const c = casesFor('mixtureLogisticResponsibilities')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const params = c['params'] as LogisticMixtureParams;
    const expected = c['expected'] as number[][];

    const actual = mixtureLogisticResponsibilities(design, targets, params);
    expect(actual).toHaveLength(expected.length);
    actual.forEach((row, n) => {
      row.forEach((v, k) => assertClose(v, expected[n]![k]!));
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 9);
    });
  });
});

describe('mixtureLogisticLogLikelihood', () => {
  it('matches PRML 14.46 computed in log space (scipy expit reference)', () => {
    const c = casesFor('mixtureLogisticLogLikelihood')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const params = c['params'] as LogisticMixtureParams;
    assertClose(mixtureLogisticLogLikelihood(design, targets, params), c['expected'] as number);
  });
});

describe('mixtureLogisticMStep', () => {
  it('matches an independent per-component Newton-Raphson solve and column-mean mixing (PRML 14.50-14.52)', () => {
    const c = casesFor('mixtureLogisticMStep')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const responsibilities = c['responsibilities'] as number[][];
    const expected = c['expected'] as LogisticMixtureParams;

    const actual = mixtureLogisticMStep(design, targets, responsibilities);
    assertParamsClose(actual, expected);
  });
});

describe('mixtureLogisticFitEM', () => {
  it('reproduces a hand-rolled E-step-then-M-step iterate trace', () => {
    const c = casesFor('mixtureLogisticEmTrace')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const initialParams = c['initialParams'] as LogisticMixtureParams;
    const steps = c['steps'] as number;
    const expectedTrace = c['expectedTrace'] as Array<{
      responsibilities: number[][];
      logLikelihoodBeforeMStep: number;
      paramsAfterMStep: LogisticMixtureParams;
    }>;

    const fit = mixtureLogisticFitEM(design, targets, initialParams, steps);

    expectedTrace.forEach((expected, i) => {
      fit.responsibilitiesHistory[i]!.forEach((row, n) =>
        row.forEach((v, k) => assertClose(v, expected.responsibilities[n]![k]!)),
      );
      assertClose(fit.logLikelihoodHistory[i]!, expected.logLikelihoodBeforeMStep);
      assertParamsClose(fit.paramsHistory[i + 1]!, expected.paramsAfterMStep);
    });
  });

  it('never decreases the log-likelihood from one iterate to the next', () => {
    const c = casesFor('mixtureLogisticEmTrace')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const initialParams = c['initialParams'] as LogisticMixtureParams;
    const steps = c['steps'] as number;

    const fit = mixtureLogisticFitEM(design, targets, initialParams, steps);
    for (let i = 1; i < fit.logLikelihoodHistory.length; i++) {
      expect(fit.logLikelihoodHistory[i]!).toBeGreaterThanOrEqual(fit.logLikelihoodHistory[i - 1]! - 1e-9);
    }
  });
});
