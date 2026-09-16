import { useMemo, useState } from 'react';
import {
  evalGrid,
  gmmFitEM,
  gmmInit,
  kmeansFit,
  kmeansInit,
  linspace,
  pcg32,
  standardNormal,
  vbGmmFit,
  vbGmmPredictiveLogPdf,
  type GaussianWishart,
  type Mat,
  type VbGmmPrior,
} from '@prml/math';
import { Axes, Bars, ContourField, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../widgets.css';

export const title = 'Automatic pruning: six components offered, two kept';
export const caption =
  'Step through the fit. Watch the bar for each surplus component sink to its prior share while EM, given the same six components, keeps every one alive.';
export const figure = '10.6';

const SEED = 20260917;
const TRUE_CLUSTERS: readonly (readonly [number, number])[] = [
  [-2.5, -2],
  [2.5, -1.5],
  [0, 2.8],
];
const POINTS_PER_CLUSTER = 18;
const K = 6;
const ROUNDS = 12;
const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 48);

function syntheticData(): Mat {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  for (const [cx, cy] of TRUE_CLUSTERS) {
    for (let i = 0; i < POINTS_PER_CLUSTER; i++) {
      points.push([cx + 0.7 * standardNormal(rng), cy + 0.7 * standardNormal(rng)]);
    }
  }
  return points;
}

const DATA = syntheticData();
const PRIOR: VbGmmPrior = { alpha0: 1e-3, beta0: 1, mean0: [0, 0], scale0: [[0.02, 0], [0, 0.02]], dof0: 2 };

export default function VbGmmPruning() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const { vbTrajectory, emAlpha } = useMemo(() => {
    const rng = pcg32(SEED + 1);
    const means = kmeansInit(rng, DATA, K);
    const initial = {
      alpha: new Array(K).fill(PRIOR.alpha0),
      components: means.map((m): GaussianWishart => ({ beta: PRIOR.beta0, mean: m, scale: PRIOR.scale0, dof: PRIOR.dof0 })),
    };
    const fit = vbGmmFit(DATA, initial, PRIOR, ROUNDS);

    const kmeans = kmeansFit(DATA, means, 1);
    const emInit = gmmInit(DATA, kmeans.meansHistory[0]!, kmeans.assignmentsHistory[0]!, K);
    const emFit = gmmFitEM(DATA, emInit, ROUNDS);
    const finalEm = emFit.paramsHistory[emFit.paramsHistory.length - 1]!;

    return { vbTrajectory: fit.posteriorHistory, emAlpha: finalEm.components.map((c) => c.weight) };
  }, []);

  const clampedStep = Math.min(step, vbTrajectory.length - 1);
  const posterior = vbTrajectory[clampedStep]!;
  const alphaTotal = posterior.alpha.reduce((s, a) => s + a, 0);
  const weights = posterior.alpha.map((a) => a / alphaTotal);

  const density = useMemo(
    () => evalGrid(GRID, GRID, (x, y) => Math.exp(vbGmmPredictiveLogPdf([x, y], posterior))),
    [posterior],
  );
  const fill = useMemo(() => {
    let peak = 0;
    for (const row of density.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak]);
  }, [density]);

  const bars = weights.map((w, k) => ({ at: k, value: w, color: w > 0.05 ? tokens.series[k % tokens.series.length]! : tokens.color.inkFaint }));
  const emBars = emAlpha.map((w, k) => ({ at: k, value: w, color: tokens.color.inkMuted }));

  return (
    <div>
      <div className="widget-grid">
        <Plot width={280} height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Predictive density under the current variational posterior">
          <ContourField data={density} levelCount={6} color={tokens.color.inkFaint} fill={fill} z={-2} opacity={0.6} />
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
          <ScatterField points={DATA.map((p) => ({ x: p[0]!, y: p[1]!, color: tokens.color.ink, size: 3 }))} />
        </Plot>
        <Plot width={280} height={220} xDomain={[-0.5, K - 0.5]} yDomain={[0, 1]} label="E[pi_k] per component, variational vs EM">
          <Axes x={{ label: 'component', ticks: [0, 1, 2, 3, 4, 5] }} y={{ label: 'weight' }} grid />
          <Bars bars={bars} thickness={0.35} />
          <Bars bars={emBars.map((b) => ({ ...b, at: b.at + 0.4 }))} thickness={0.35} />
        </Plot>
      </div>
      <StepThrough step={clampedStep} stepCount={vbTrajectory.length} onStep={setStep} labels={vbTrajectory.map((_, i) => `sweep ${i}`)} />
      <p className="widget-readout">
        {weights.filter((w) => w > 0.05).length} of {K} components hold non-negligible weight under the variational
        posterior; EM (grey bars, converged) still uses all {K}, because nothing in its objective rewards deleting one.
      </p>
    </div>
  );
}
