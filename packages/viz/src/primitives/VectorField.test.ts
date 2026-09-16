import { describe, expect, it } from 'vitest';

import { computeScaledArrows } from './VectorField.js';

const identityToPx = (x: number, y: number): readonly [number, number] => [x, y];

describe('computeScaledArrows', () => {
  it('scales every arrow by one global factor so the longest hits maxLength', () => {
    const origins = [
      [0, 0],
      [10, 10],
    ] as const;
    const field = (x: number, _y: number): readonly [number, number] => (x === 0 ? [1, 0] : [2, 0]);

    const arrows = computeScaledArrows(origins, field, identityToPx, 20);

    expect(arrows).toHaveLength(2);
    expect(Math.hypot(arrows[1]!.dxPx, arrows[1]!.dyPx)).toBeCloseTo(20, 5);
    expect(Math.hypot(arrows[0]!.dxPx, arrows[0]!.dyPx)).toBeCloseTo(10, 5);
  });

  it('preserves relative magnitude rather than normalising each arrow to maxLength', () => {
    const origins = [[0, 0]] as const;
    const field = (): readonly [number, number] => [3, 4];

    const arrows = computeScaledArrows(origins, field, identityToPx, 10);

    expect(arrows[0]!.dxPx).toBeCloseTo(6, 5);
    expect(arrows[0]!.dyPx).toBeCloseTo(8, 5);
  });

  it('accounts for anisotropic pixel scaling when finding the longest arrow', () => {
    const origins = [
      [0, 0],
      [5, 5],
    ] as const;
    const field = (x: number, _y: number): readonly [number, number] => (x === 0 ? [1, 0] : [0, 1]);
    const anisotropicToPx = (x: number, y: number): readonly [number, number] => [x * 10, y * 1];

    const arrows = computeScaledArrows(origins, field, anisotropicToPx, 100);

    expect(Math.hypot(arrows[0]!.dxPx, arrows[0]!.dyPx)).toBeCloseTo(100, 5);
    expect(Math.hypot(arrows[1]!.dxPx, arrows[1]!.dyPx)).toBeCloseTo(10, 5);
  });

  it('reports the raw data-space magnitude for colouring, independent of pixel scale', () => {
    const origins = [[0, 0]] as const;
    const field = (): readonly [number, number] => [3, 4];
    const arrows = computeScaledArrows(origins, field, identityToPx, 10);
    expect(arrows[0]!.magnitude).toBeCloseTo(5, 5);
  });

  it('does not divide by zero when every vector is zero', () => {
    const origins = [[0, 0]] as const;
    const field = (): readonly [number, number] => [0, 0];
    const arrows = computeScaledArrows(origins, field, identityToPx, 10);
    expect(arrows[0]!.dxPx).toBe(0);
    expect(arrows[0]!.dyPx).toBe(0);
  });
});
