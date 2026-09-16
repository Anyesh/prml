import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { trainToyNetworks } from './mdnTraining.js';

const WEIGHTS_PATH = path.join(import.meta.dirname, 'mdnToyWeights.json');

describe('the committed mixture density network weights', () => {
  it('still match what the training code produces', () => {
    const committed = JSON.parse(fs.readFileSync(WEIGHTS_PATH, 'utf8')) as {
      regression: number[];
      mdn: number[];
    };
    const trained = trainToyNetworks();

    expect(trained.regression).toHaveLength(committed.regression.length);
    expect(trained.mdn).toHaveLength(committed.mdn.length);
    trained.regression.forEach((w, i) => expect(w).toBeCloseTo(committed.regression[i]!, 12));
    trained.mdn.forEach((w, i) => expect(w).toBeCloseTo(committed.mdn[i]!, 12));
  });
});
