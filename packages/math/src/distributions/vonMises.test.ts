import { describe, it, expect } from 'vitest';
import { loadFixture, closeTo } from '../testing/fixture.js';
import { pcg32 } from '../rng.js';
import { vonMisesLogPdf, vonMisesPdf, vonMisesSample, vonMisesFit, type VonMisesParams } from './vonMises.js';

interface PdfCase {
  readonly fn: 'logPdf' | 'pdf';
  readonly theta: number;
  readonly params: VonMisesParams;
  readonly expected: number;
}

interface FitCase {
  readonly fn: 'fit';
  readonly theta: number[];
  readonly expected: { mu: number; kappa: number };
}

const fixture = loadFixture<{ cases: (PdfCase | FitCase)[] }>('vonMises');

function casesFor<T>(fn: string): T[] {
  return fixture.cases.filter((c) => c.fn === fn) as T[];
}

describe('vonMisesLogPdf', () => {
  it('matches scipy on every golden case, including kappa = 0 and kappa = 700', () => {
    const cases = casesFor<PdfCase>('logPdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = vonMisesLogPdf(c.theta, c.params);
      expect(closeTo(actual, c.expected), `logPdf(${c.theta}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('vonMisesPdf', () => {
  it('matches scipy on every golden case', () => {
    const cases = casesFor<PdfCase>('pdf');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = vonMisesPdf(c.theta, c.params);
      expect(closeTo(actual, c.expected), `pdf(${c.theta}, ${JSON.stringify(c.params)})`).toBe(true);
    }
  });
});

describe('vonMisesFit', () => {
  it('recovers mu and kappa by inverting A(kappa) = I1(kappa)/I0(kappa), matching an independent scipy brentq solve', () => {
    const cases = casesFor<FitCase>('fit');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const actual = vonMisesFit(c.theta);
      // kappa near 0 is flat and noise-sensitive; a looser absolute tolerance is used there,
      // 1e-9 relative everywhere kappa is not degenerate.
      const kappaTol = c.expected.kappa < 1 ? 1e-6 : 1e-9;
      expect(closeTo(actual.mu, c.expected.mu), `fit(...).mu for n=${c.theta.length}`).toBe(true);
      expect(closeTo(actual.kappa, c.expected.kappa, kappaTol), `fit(...).kappa for n=${c.theta.length}`).toBe(true);
    }
  });
});

describe('vonMisesSample', () => {
  it('wraps to (-pi, pi] and concentrates around mu', () => {
    const rng = pcg32(31415);
    const p: VonMisesParams = { mu: 1.0, kappa: 8 };
    const n = 20000;
    let cosSum = 0;
    let sinSum = 0;
    for (let i = 0; i < n; i++) {
      const theta = vonMisesSample(rng, p);
      expect(theta).toBeGreaterThan(-Math.PI);
      expect(theta).toBeLessThanOrEqual(Math.PI);
      cosSum += Math.cos(theta);
      sinSum += Math.sin(theta);
    }
    const meanAngle = Math.atan2(sinSum / n, cosSum / n);
    expect(Math.abs(meanAngle - p.mu)).toBeLessThan(0.05);
  });

  it('is uniform when kappa = 0', () => {
    const rng = pcg32(2718);
    const p: VonMisesParams = { mu: 0, kappa: 0 };
    let cosSum = 0;
    let sinSum = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
      const theta = vonMisesSample(rng, p);
      cosSum += Math.cos(theta);
      sinSum += Math.sin(theta);
    }
    expect(Math.abs(cosSum / n)).toBeLessThan(0.03);
    expect(Math.abs(sinSum / n)).toBeLessThan(0.03);
  });
});
