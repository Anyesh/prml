import { describe, expect, it } from 'vitest';
import { sigmoid } from '../numeric.js';
import { loadFixture } from '../testing/fixture.js';
import {
  logisticLambda,
  logisticLocalBound,
  updateXi,
  variationalLogisticFit,
  variationalLogisticLowerBound,
  variationalLogisticUpdate,
  updateLogisticAlpha,
  type VariationalLogisticPosterior,
  type VariationalLogisticPrior,
} from './logisticBound.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('logisticBound');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('logisticLambda', () => {
  it('matches (sigmoid(xi) - 0.5) / (2 xi), with the removable singularity at xi = 0 filled by its limit 1/8', () => {
    const c = casesFor('logisticLambda')[0]!;
    const xi = c['xi'] as number[];
    const expected = c['expected'] as number[];
    xi.forEach((v, i) => expect(logisticLambda(v)).toBeCloseTo(expected[i]!, 9));
  });

  it('is symmetric in xi, since the bound must be too', () => {
    for (const xi of [0.3, 1.7, 4.2]) {
      expect(logisticLambda(xi)).toBeCloseTo(logisticLambda(-xi), 12);
    }
  });
});

describe('logisticLocalBound', () => {
  it('matches sigma(xi) exp((x - xi)/2 - lambda(xi)(x^2 - xi^2)) (PRML 10.144)', () => {
    for (const c of casesFor('logisticLocalBound')) {
      const x = c['x'] as number[];
      const xi = c['xi'] as number;
      const expected = c['expected'] as number[];
      x.forEach((xv, i) => expect(logisticLocalBound(xv, xi)).toBeCloseTo(expected[i]!, 9));
    }
  });

  it('touches the true sigmoid exactly at x = xi and x = -xi', () => {
    for (const c of casesFor('logisticLocalBound_touchPoints')) {
      const xi = c['xi'] as number;
      const [xPlus, xMinus] = c['x'] as [number, number];
      const [expectedPlus, expectedMinus] = c['expected'] as [number, number];
      expect(logisticLocalBound(xPlus, xi)).toBeCloseTo(expectedPlus, 9);
      expect(logisticLocalBound(xMinus, xi)).toBeCloseTo(expectedMinus, 9);
    }
  });

  it('never exceeds the true sigmoid anywhere, being a lower bound', () => {
    for (const xi of [0.5, 1.5, 3.0]) {
      for (let x = -10; x <= 10; x += 0.37) {
        expect(logisticLocalBound(x, xi)).toBeLessThanOrEqual(sigmoid(x) + 1e-12);
      }
    }
  });
});

describe('variationalLogisticUpdate', () => {
  it('matches the closed-form q(w) given fixed xi (PRML 10.157-10.159)', () => {
    const c = casesFor('variationalLogisticUpdate')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const prior = c['prior'] as VariationalLogisticPrior;
    const xi = c['xi'] as number[];
    const expected = c['expected'] as { mean: number[]; cov: number[][] };
    const out = variationalLogisticUpdate(design, targets, prior, xi);
    out.mean.forEach((v, i) => expect(v).toBeCloseTo(expected.mean[i]!, 9));
    out.cov.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected.cov[i]![j]!, 9)));
  });
});

describe('updateXi', () => {
  it('matches xi_n^2 = phi_n^T (S + m m^T) phi_n (PRML 10.163-10.164)', () => {
    const c = casesFor('updateXi')[0]!;
    const design = c['design'] as number[][];
    const mean = c['mean'] as number[];
    const cov = c['cov'] as number[][];
    const expected = c['expected'] as number[];
    const out = updateXi(design, mean, cov);
    out.forEach((v, i) => expect(v).toBeCloseTo(expected[i]!, 9));
  });
});

describe('variationalLogisticLowerBound', () => {
  it('matches the closed-form bound (PRML 10.161)', () => {
    const c = casesFor('variationalLogisticLowerBound')[0]!;
    const prior = c['prior'] as VariationalLogisticPrior;
    const posterior = c['posterior'] as { mean: number[]; cov: number[][] };
    const xi = c['xi'] as number[];
    expect(variationalLogisticLowerBound(prior, posterior, xi)).toBeCloseTo(c['expected'] as number, 9);
  });
});

describe('variationalLogisticFit', () => {
  it('reproduces an independently computed multi-round trace', () => {
    const c = casesFor('variationalLogisticFit')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const prior = c['prior'] as VariationalLogisticPrior;
    const xiInit = c['xiInit'] as number[];
    const rounds = c['rounds'] as number;
    const expectedTrace = c['expectedTrace'] as { mean: number[]; cov: number[][]; xi: number[]; lowerBound: number }[];

    const result = variationalLogisticFit(design, targets, prior, xiInit, rounds);
    expect(result.posteriorHistory).toHaveLength(expectedTrace.length);
    result.posteriorHistory.forEach((p, i) => {
      const e = expectedTrace[i]!;
      p.mean.forEach((v, k) => expect(v).toBeCloseTo(e.mean[k]!, 9));
      p.cov.forEach((row, a) => row.forEach((v, b) => expect(v).toBeCloseTo(e.cov[a]![b]!, 9)));
    });
    result.xiHistory.forEach((xi, i) => xi.forEach((v, k) => expect(v).toBeCloseTo(expectedTrace[i]!.xi[k]!, 9)));
    result.lowerBoundHistory.forEach((lb, i) => expect(lb).toBeCloseTo(expectedTrace[i]!.lowerBound, 9));
  });

  it('the lower bound rises monotonically across rounds', () => {
    const c = casesFor('variationalLogisticFit')[0]!;
    const design = c['design'] as number[][];
    const targets = c['targets'] as number[];
    const prior = c['prior'] as VariationalLogisticPrior;
    const xiInit = c['xiInit'] as number[];
    const rounds = c['rounds'] as number;
    const result = variationalLogisticFit(design, targets, prior, xiInit, rounds);
    for (let i = 1; i < result.lowerBoundHistory.length; i++) {
      expect(result.lowerBoundHistory[i]!).toBeGreaterThanOrEqual(result.lowerBoundHistory[i - 1]! - 1e-9);
    }
  });
});

describe('updateLogisticAlpha', () => {
  it('matches the evidence-style point estimate gamma / (mean^T mean) (PRML 3.92 applied to the variational posterior)', () => {
    const c = casesFor('updateLogisticAlpha')[0]!;
    const posterior = c['posterior'] as VariationalLogisticPosterior;
    const alpha = c['alpha'] as number;
    expect(updateLogisticAlpha(posterior, alpha)).toBeCloseTo(c['expected'] as number, 9);
  });
});
