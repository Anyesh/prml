import { useMemo, useState } from 'react';
import { icmDenoise, linspace, pcg32, type Grid } from '@prml/math';
import { categoricalScale, Heatmap, Plot, useResolvedTokens } from '@prml/viz';
import { Slider, StepThrough } from '@prml/ui';
import '../widgets.css';

const SIZE = 20;
const CENTER = (SIZE - 1) / 2;
const NOISE_FRACTION = 0.08;
const MAX_SWEEPS = 8;
const ETA = 2.1;

function buildClean(): Grid {
  const rows: number[][] = [];
  for (let i = 0; i < SIZE; i++) {
    const row: number[] = [];
    for (let j = 0; j < SIZE; j++) {
      row.push(Math.abs(i - CENTER) <= 2 || Math.abs(j - CENTER) <= 2 ? 1 : -1);
    }
    rows.push(row);
  }
  return rows;
}

function buildNoisy(clean: Grid): Grid {
  const rng = pcg32(20260914);
  return clean.map((row) =>
    row.map((v) => (rng.next() < NOISE_FRACTION ? -v : v)),
  );
}

const CLEAN = buildClean();
const NOISY = buildNoisy(CLEAN);
const XS = linspace(0, SIZE - 1, SIZE);
const YS = linspace(0, SIZE - 1, SIZE);

function flippedCount(a: Grid, b: Grid): number {
  let count = 0;
  a.forEach((row, i) => row.forEach((v, j) => { if (v !== b[i]![j]) count += 1; }));
  return count;
}

export default function ImageDenoisingICM() {
  const [beta, setBeta] = useState(1.0);
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();
  const colorAt = categoricalScale([tokens.color.plotBg, tokens.color.ink]);

  const result = useMemo(() => icmDenoise(NOISY, NOISY, { h: 0, beta, eta: ETA }, MAX_SWEEPS), [beta]);
  const clampedStep = Math.min(step, result.history.length - 1);
  const current = result.history[clampedStep]!;
  const converged = clampedStep > 0 && flippedCount(current, result.history[clampedStep - 1]!) === 0;
  const stillNoisy = flippedCount(current, NOISY);
  const stillWrong = flippedCount(current, CLEAN);

  const labels = result.history.map((_, i) => (i === 0 ? 'Noisy input' : `After sweep ${i}`));

  return (
    <div>
      <div className="widget-grid">
        <Plot width={260} height={260} xDomain={[0, SIZE - 1]} yDomain={[0, SIZE - 1]} equalAspect label="Noisy input image, fixed throughout">
          <Heatmap data={{ xs: XS, ys: YS, values: NOISY }} interpolator={colorAt} />
        </Plot>
        <Plot width={260} height={260} xDomain={[0, SIZE - 1]} yDomain={[0, SIZE - 1]} equalAspect label="Current de-noised image">
          <Heatmap data={{ xs: XS, ys: YS, values: current }} interpolator={colorAt} />
        </Plot>
      </div>
      <Slider
        label="Coupling strength (beta)"
        value={beta}
        onChange={setBeta}
        min={0.1}
        max={3}
        step={0.05}
        hint="Higher beta pulls each pixel harder towards its neighbours; past a point the plus shape over-smooths into a blob."
      />
      <StepThrough step={clampedStep} stepCount={result.history.length} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        Energy {result.energyHistory[clampedStep]!.toFixed(1)}, {stillNoisy} of {SIZE * SIZE} pixels still differ from
        the noisy input, {stillWrong} still differ from the clean plus shape.
        {converged ? ' No pixel changed on the last sweep: ICM has converged to a local energy minimum.' : ''}
      </p>
    </div>
  );
}
