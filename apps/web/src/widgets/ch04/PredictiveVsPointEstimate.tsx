import { useMemo, useState } from 'react';
import { evalGrid, fitLaplaceLogisticPosterior, laplaceLogisticPredictive, linspace, logisticPredict, pcg32, standardNormal } from '@prml/math';
import { Axes, ContourField, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const N_PER_CLASS = 10;
const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(-6, 6, 70);
const LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9];

const rng = pcg32(20260440);
const CLASS0 = Array.from({ length: N_PER_CLASS }, () => [1, -1.5 + standardNormal(rng), standardNormal(rng)]);
const CLASS1 = Array.from({ length: N_PER_CLASS }, () => [1, 1.5 + standardNormal(rng), standardNormal(rng)]);
const DESIGN = [...CLASS0, ...CLASS1];
const TARGETS = [...CLASS0.map(() => 0), ...CLASS1.map(() => 1)];

export default function PredictiveVsPointEstimate() {
  const [priorVariance, setPriorVariance] = useState(4);
  const tokens = useResolvedTokens();

  const posterior = useMemo(
    () =>
      fitLaplaceLogisticPosterior(DESIGN, TARGETS, {
        mean: [0, 0, 0],
        covariance: [
          [priorVariance, 0, 0],
          [0, priorVariance, 0],
          [0, 0, priorVariance],
        ],
      }),
    [priorVariance],
  );

  const pointField = useMemo(() => evalGrid(GRID, GRID, (x, y) => logisticPredict(posterior.mean, [1, x, y])), [posterior]);
  const predictiveField = useMemo(
    () => evalGrid(GRID, GRID, (x, y) => laplaceLogisticPredictive([1, x, y], posterior)),
    [posterior],
  );

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Point-estimate probability contours against the Laplace-marginalised predictive">
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
        <ContourField data={pointField} levels={LEVELS} color={tokens.color.danger} lineWidth={1.25} />
        <ContourField data={predictiveField} levels={LEVELS} color={tokens.color.accent} lineWidth={1.25} />
        <ScatterField points={CLASS0.map((p, i) => ({ x: p[1]!, y: p[2]!, id: `0-${i}` }))} color={tokens.series[0]} size={3.5} />
        <ScatterField points={CLASS1.map((p, i) => ({ x: p[1]!, y: p[2]!, id: `1-${i}` }))} color={tokens.series[1]} size={3.5} />
        <Legend
          entries={[
            { label: 'σ(wMAPᵀφ), point estimate', color: tokens.color.danger, mark: 'line' },
            { label: 'marginalised predictive', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <Slider
        label="Prior variance"
        value={priorVariance}
        onChange={setPriorVariance}
        min={0.2}
        max={20}
        scale="log"
        hint="The 0.5 contours always coincide. Watch the 0.1 and 0.9 contours pull back toward the data as this grows: away from the data, the posterior is wide, and the predictive probability gets pulled toward one half."
      />
    </div>
  );
}
