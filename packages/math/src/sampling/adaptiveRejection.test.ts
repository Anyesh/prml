import { describe, expect, it } from 'vitest';
import { pcg32 } from '../rng.js';
import { closeTo, loadFixture } from '../testing/fixture.js';
import { arsBuildEnvelope, arsPieceArea, arsPieceQuantile, arsSample, type ArsPoint } from './adaptiveRejection.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('sampling_adaptive_rejection');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

const LOG_2PI = Math.log(2 * Math.PI);
function h(z: number): number {
  return -0.5 * z * z - 0.5 * LOG_2PI;
}
function hPrime(z: number): number {
  return -z;
}

describe('arsBuildEnvelope', () => {
  it('matches independently computed tangent-line breakpoints for a standard normal', () => {
    const c = casesFor('arsBuildEnvelope')[0]!;
    const points = c['points'] as ArsPoint[];
    const domain = c['domain'] as [number, number];
    const envelope = arsBuildEnvelope(points, domain);
    const expected = (c['expected'] as { pieces: { lo: number; hi: number; z: number; h: number; hPrime: number }[] }).pieces;
    expect(envelope.pieces).toHaveLength(expected.length);
    envelope.pieces.forEach((p, i) => {
      expect(closeTo(p.lo, expected[i]!.lo)).toBe(true);
      expect(closeTo(p.hi, expected[i]!.hi)).toBe(true);
      expect(closeTo(p.z, expected[i]!.z)).toBe(true);
    });
  });
});

describe('arsPieceArea / arsPieceQuantile', () => {
  it('matches an independent numerical integration of each piece area', () => {
    const c0 = casesFor('arsBuildEnvelope')[0]!;
    const points = c0['points'] as ArsPoint[];
    const domain = c0['domain'] as [number, number];
    const envelope = arsBuildEnvelope(points, domain);

    for (const c of casesFor('arsPieceArea')) {
      const piece = envelope.pieces[c['pieceIndex'] as number]!;
      const area = arsPieceArea(piece, c['shift'] as number);
      expect(closeTo(area, c['expected'] as number)).toBe(true);
    }
  });

  it('matches an independent numerical integration for the envelope median of each piece', () => {
    const c0 = casesFor('arsBuildEnvelope')[0]!;
    const points = c0['points'] as ArsPoint[];
    const domain = c0['domain'] as [number, number];
    const envelope = arsBuildEnvelope(points, domain);

    for (const c of casesFor('arsPieceQuantile')) {
      const piece = envelope.pieces[c['pieceIndex'] as number]!;
      const z = arsPieceQuantile(piece, c['shift'] as number, c['t'] as number);
      expect(closeTo(z, c['expected'] as number)).toBe(true);
    }
  });
});

describe('arsSample', () => {
  it('reproduces a reference draw-test-refine trajectory bit-for-bit for a fixed seed', () => {
    const cases = casesFor('arsSample');
    expect(cases.length).toBeGreaterThan(0);
    for (const c of cases) {
      const rng = pcg32(c['seed'] as number, 1);
      const grid = c['grid'] as number[];
      const domain = c['domain'] as [number, number];
      const result = arsSample(rng, { logTarget: h, dLogTarget: hPrime, initialGrid: grid, domain });
      const expected = c['expected'] as { sample: number; trace: unknown[]; finalPieceCount: number };
      expect(closeTo(result.sample, expected.sample)).toBe(true);
      expect(result.attempts).toBe(expected.trace.length);
      expect(result.envelope.pieces.length).toBe(expected.finalPieceCount);
    }
  });

  it('accepted samples track the standard normal mean and variance', () => {
    const rng = pcg32(909, 4);
    const n = 8000;
    let sum = 0;
    let sumSq = 0;
    let envelope;
    for (let i = 0; i < n; i++) {
      const result = arsSample(
        rng,
        { logTarget: h, dLogTarget: hPrime, initialGrid: [-2, -0.5, 0, 1, 3], domain: [-6, 6] },
        envelope,
      );
      envelope = result.envelope;
      sum += result.sample;
      sumSq += result.sample * result.sample;
    }
    const mean = sum / n;
    const variance = sumSq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(variance - 1)).toBeLessThan(0.08);
  });
});
