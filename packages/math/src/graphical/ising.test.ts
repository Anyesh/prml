import { describe, expect, it } from 'vitest';
import { loadFixture } from '../testing/fixture.js';
import { icmDenoise, icmSweep, isingEnergy, type Grid, type IsingParams } from './ising.js';

interface Case {
  readonly fn: string;
  readonly [key: string]: unknown;
}

const fixture = loadFixture<{ cases: Case[] }>('isingIcm');
function casesFor(fn: string): Case[] {
  return fixture.cases.filter((c) => c.fn === fn);
}

describe('isingEnergy', () => {
  it('matches PRML 8.42 on a clean and a noisy image, with and without a bias field', () => {
    for (const c of [...casesFor('isingEnergy'), ...casesFor('isingEnergy_noisy')]) {
      const x = c['x'] as Grid;
      const y = c['y'] as Grid;
      const params = c['params'] as IsingParams;
      expect(isingEnergy(x, y, params)).toBeCloseTo(c['expected'] as number, 9);
    }
  });
});

describe('icmSweep', () => {
  it('reproduces one raster-order sweep, each pixel set by the current values of its neighbours', () => {
    const c = casesFor('icmSweep')[0]!;
    const x = c['x'] as Grid;
    const y = c['y'] as Grid;
    const params = c['params'] as IsingParams;
    const expected = c['expected'] as Grid;
    const out = icmSweep(x, y, params);
    out.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expected[i]![j]!, 9)));
  });
});

describe('icmDenoise', () => {
  it('reproduces the full sweep-by-sweep trace and its energy history', () => {
    const c = casesFor('icmDenoise')[0]!;
    const initialX = c['initialX'] as Grid;
    const y = c['y'] as Grid;
    const params = c['params'] as IsingParams;
    const sweeps = c['sweeps'] as number;
    const expectedHistory = c['expectedHistory'] as Grid[];
    const expectedEnergyHistory = c['expectedEnergyHistory'] as number[];

    const result = icmDenoise(initialX, y, params, sweeps);

    expect(result.history).toHaveLength(expectedHistory.length);
    result.history.forEach((grid, s) =>
      grid.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(expectedHistory[s]![i]![j]!, 9))),
    );
    result.energyHistory.forEach((e, s) => expect(e).toBeCloseTo(expectedEnergyHistory[s]!, 9));
  });

  it('never increases the energy from one sweep to the next, the guarantee ICM relies on', () => {
    const c = casesFor('icmDenoise')[0]!;
    const initialX = c['initialX'] as Grid;
    const y = c['y'] as Grid;
    const params = c['params'] as IsingParams;
    const result = icmDenoise(initialX, y, params, 6);
    for (let i = 1; i < result.energyHistory.length; i++) {
      expect(result.energyHistory[i]!).toBeLessThanOrEqual(result.energyHistory[i - 1]! + 1e-9);
    }
  });
});
