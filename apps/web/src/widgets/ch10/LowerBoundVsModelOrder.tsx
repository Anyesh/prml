import { useMemo, useState } from 'react';
import { designMatrix, pcg32, polynomialBasis, standardNormal, vlrFit } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'The lower bound picks the true model order';
export const caption =
  'Drag the noise level up. The bound keeps peaking at the order that generated the data, even as the data gets noisier and a maximum-likelihood fit would keep favouring more terms.';
export const figure = '10.9';

const TRUE_ORDER = 3;
const CANDIDATE_ORDERS = [1, 2, 3, 4, 5, 6, 7];
const N = 12;
const SEED = 20260917;
const TRUE_WEIGHTS = [0.3, 1.2, -0.8, 0.5];
const PRIOR = { a0: 1e-3, b0: 1e-3 };
const ITERS = 30;

function syntheticTargets(noiseScale: number): { xs: number[]; targets: number[] } {
  const rng = pcg32(SEED);
  const xs = Array.from({ length: N }, (_, i) => -2 + (4 * i) / (N - 1));
  const targets = xs.map((x) => {
    const truePhi = polynomialBasis(TRUE_ORDER)(x);
    const mean = truePhi.reduce((s, v, i) => s + v * TRUE_WEIGHTS[i]!, 0);
    return mean + noiseScale * standardNormal(rng);
  });
  return { xs, targets };
}

function boundAtOrder(order: number, xs: number[], targets: number[], beta: number): number {
  const design = designMatrix(xs, polynomialBasis(order));
  const fit = vlrFit(design, targets, beta, PRIOR, ITERS);
  return fit.lowerBoundHistory[fit.lowerBoundHistory.length - 1]!;
}

export default function LowerBoundVsModelOrder() {
  const [noiseScale, setNoiseScale] = useState(0.3);
  const tokens = useResolvedTokens();

  const { xs, targets } = useMemo(() => syntheticTargets(noiseScale), [noiseScale]);
  const beta = 1 / (noiseScale * noiseScale);
  const bounds = useMemo(() => CANDIDATE_ORDERS.map((order) => boundAtOrder(order, xs, targets, beta)), [xs, targets, beta]);
  const best = Math.max(...bounds);
  const bestOrder = CANDIDATE_ORDERS[bounds.indexOf(best)]!;

  const bars = CANDIDATE_ORDERS.map((order, i) => ({
    at: order,
    value: bounds[i]!,
    color: order === bestOrder ? tokens.color.accent : tokens.color.inkMuted,
  }));
  const range = Math.max(...bounds) - Math.min(...bounds) || 1;

  return (
    <div>
      <Plot height={220} xDomain={[0.5, 7.5]} yDomain={[Math.min(...bounds) - 0.05 * range, best + 0.05 * range]} label="Converged lower bound as a function of polynomial order">
        <Axes x={{ label: 'polynomial order M', ticks: CANDIDATE_ORDERS }} y={{ label: 'L (lower bound)' }} grid />
        <Bars bars={bars} thickness={0.5} baseline={Math.min(...bounds) - 0.05 * range} />
      </Plot>
      <Slider
        label="observation noise (1/sqrt(beta))"
        value={noiseScale}
        onChange={setNoiseScale}
        min={0.1}
        max={1.2}
        step={0.02}
        format={(v) => v.toFixed(2)}
        hint="The data always comes from a cubic. Push the noise up and see how long the bound keeps finding it."
      />
      <p className="widget-readout">
        True order is {TRUE_ORDER}; the bound currently peaks at order {bestOrder}.
      </p>
    </div>
  );
}
