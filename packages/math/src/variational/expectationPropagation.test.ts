import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import {
  clutterEpFit,
  clutterEpInit,
  clutterMomentMatch,
  isotropicCavity,
  refineIsotropicSite,
  type ClutterEpState,
  type ClutterModel,
  type ClutterMomentMatch,
  type IsotropicGaussian,
  type IsotropicSite,
} from './expectationPropagation.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('expectationPropagation');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

function expectGaussianClose(actual: IsotropicGaussian, expected: IsotropicGaussian, digits = 9) {
  actual.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, digits));
  expect(actual.variance).toBeCloseTo(expected.variance, digits);
}

function expectSiteClose(actual: IsotropicSite, expected: IsotropicSite, digits = 9) {
  expect(actual.logScale).toBeCloseTo(expected.logScale, digits);
  actual.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, digits));
  expect(actual.variance).toBeCloseTo(expected.variance, digits);
}

describe('isotropicCavity', () => {
  it('matches the leave-one-site-out cavity (PRML 10.214-10.215)', () => {
    const c = casesFor('isotropicCavity')[0]!;
    const out = isotropicCavity(c['q'] as IsotropicGaussian, c['site'] as IsotropicSite);
    expectGaussianClose(out, c['expected'] as IsotropicGaussian);
  });
});

describe('clutterMomentMatch', () => {
  it('matches the exact tilted-distribution moments of the two-component mixture (PRML 10.216-10.219)', () => {
    for (const c of casesFor('clutterMomentMatch')) {
      const out = clutterMomentMatch(c['x'] as number[], c['cavity'] as IsotropicGaussian, c['model'] as ClutterModel);
      const expected = c['expected'] as ClutterMomentMatch;
      expect(out.normaliser).toBeCloseTo(expected.normaliser, 9);
      expect(out.signalProbability).toBeCloseTo(expected.signalProbability, 9);
      expectGaussianClose(out.posterior, expected.posterior);
    }
  });

  it('reports a signal probability strictly between zero and one', () => {
    for (const c of casesFor('clutterMomentMatch')) {
      const out = clutterMomentMatch(c['x'] as number[], c['cavity'] as IsotropicGaussian, c['model'] as ClutterModel);
      expect(out.signalProbability).toBeGreaterThan(0);
      expect(out.signalProbability).toBeLessThan(1);
    }
  });
});

describe('refineIsotropicSite', () => {
  it('matches the site recovered from the Gaussian convolution identity (PRML 10.220-10.222)', () => {
    for (const c of casesFor('refineIsotropicSite')) {
      const out = refineIsotropicSite(c['cavity'] as IsotropicGaussian, c['moments'] as ClutterMomentMatch);
      expectSiteClose(out, c['expected'] as IsotropicSite);
    }
  });
});

describe('clutterEpFit', () => {
  it('matches a golden one-sweep trace over six points, posterior and every site', () => {
    const c = casesFor('clutterEpFit')[0]!;
    const data = c['data'] as number[][];
    const prior = c['prior'] as IsotropicGaussian;
    const model = c['model'] as ClutterModel;
    const sweeps = c['sweeps'] as number;
    const expectedTrace = c['expectedTrace'] as ClutterEpState[];

    const initial = clutterEpInit(prior, data.length);
    const history = clutterEpFit(data, initial, model, sweeps);

    expect(history).toHaveLength(sweeps + 1);
    history.slice(1).forEach((state, i) => {
      const expected = expectedTrace[i]!;
      expectGaussianClose(state.posterior, expected.posterior);
      state.sites.forEach((site, k) => expectSiteClose(site, expected.sites[k]!));
    });
  });

  it('starts every site uninformative, so the first cavity equals the prior', () => {
    const prior: IsotropicGaussian = { mean: [0, 0], variance: 50 };
    const initial = clutterEpInit(prior, 3);
    const cavity = isotropicCavity(initial.posterior, initial.sites[0]!);
    expectGaussianClose(cavity, prior, 6);
  });
});
