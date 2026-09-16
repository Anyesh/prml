import { describe, expect, it } from 'vitest';

import { computeLevels, contourPointToData, flattenField, indexToCoord } from './ContourField.js';

describe('flattenField', () => {
  it('flattens values[j][i] in row-major order without transposing a non-square grid', () => {
    const data = {
      xs: [0, 1, 2],
      ys: [0, 10],
      values: [
        [1, 2, 3],
        [4, 5, 6],
      ],
    };

    const flattened = flattenField(data);

    expect(flattened.width).toBe(3);
    expect(flattened.height).toBe(2);
    expect(flattened.values).toEqual([1, 2, 3, 4, 5, 6]);
    for (let j = 0; j < data.ys.length; j++) {
      for (let i = 0; i < data.xs.length; i++) {
        expect(flattened.values[j * flattened.width + i]).toBe(data.values[j]![i]);
      }
    }
  });
});

describe('indexToCoord', () => {
  const xs = [0, 1, 3, 7];

  it('returns exact coordinates at integer indices on an uneven grid', () => {
    expect(indexToCoord(0, xs)).toBe(0);
    expect(indexToCoord(1, xs)).toBe(1);
    expect(indexToCoord(2, xs)).toBe(3);
    expect(indexToCoord(3, xs)).toBe(7);
  });

  it('interpolates linearly between uneven neighbours', () => {
    expect(indexToCoord(1.5, xs)).toBe(2);
    expect(indexToCoord(2.5, xs)).toBe(5);
  });

  it('extrapolates past the ends using the boundary segment slope', () => {
    expect(indexToCoord(-0.5, xs)).toBe(-0.5);
    expect(indexToCoord(3.5, xs)).toBe(9);
  });
});

describe('contourPointToData', () => {
  it('removes the d3-contour half-cell offset before mapping through xs/ys', () => {
    const xs = [0, 2, 4];
    const ys = [0, 10, 20];

    expect(contourPointToData([0.5, 0.5], xs, ys)).toEqual([0, 0]);
    expect(contourPointToData([1.5, 1.5], xs, ys)).toEqual([2, 10]);
    expect(contourPointToData([2.5, 2.5], xs, ys)).toEqual([4, 20]);
  });
});

describe('computeLevels', () => {
  it('sorts and returns explicit levels unchanged', () => {
    expect(computeLevels([1, 2, 3], [5, 1, 3])).toEqual([1, 3, 5]);
  });

  it('takes levelCount interior quantiles of the value distribution', () => {
    const values = Array.from({ length: 101 }, (_, i) => i);
    const levels = computeLevels(values, undefined, 3);
    expect(levels).toHaveLength(3);
    expect(levels[0]).toBeCloseTo(25, 0);
    expect(levels[1]).toBeCloseTo(50, 0);
    expect(levels[2]).toBeCloseTo(75, 0);
  });

  it('ignores non-finite values when computing the distribution', () => {
    const values = [1, 2, NaN, Infinity, 3, -Infinity];
    const levels = computeLevels(values, undefined, 1);
    expect(levels).toHaveLength(1);
    expect(levels[0]).toBeCloseTo(2, 0);
  });

  it('returns an empty array when there is no finite data', () => {
    expect(computeLevels([NaN, Infinity], undefined, 5)).toEqual([]);
  });
});
