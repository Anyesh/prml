import { useMemo, useState } from 'react';
import {
  evalGrid,
  gmmEmStep,
  gmmPdf,
  kmeansAssign,
  kmeansFit,
  kmeansInit,
  linspace,
  pcg32,
  type GmmParams,
  type Mat,
} from '@prml/math';
import { Axes, blendColors, categoricalScale, ContourField, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { faithfulLikeData } from './data.js';

import '../widgets.css';

export const title = 'Hard versus soft: K-means and EM side by side';
export const caption =
  'Step through the same rounds on both panels. The left hardens every point to one colour at each E-step; the right blends colours by responsibility. Watch the boundary points.';
export const figure = '9.5';

const DATA = faithfulLikeData();
const K = 2;
const ROUNDS = 5;
const SEED = 20260915;
const DOMAIN: readonly [number, number] = [-4, 4];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 56);
const ISOTROPIC_COV = [
  [0.8, 0],
  [0, 0.8],
];

interface KMeansFrame {
  readonly kind: 'init' | 'E' | 'M';
  readonly round: number;
  readonly means: Mat;
  readonly assignments: readonly number[] | null;
}

interface EmFrame {
  readonly kind: 'init' | 'E' | 'M';
  readonly round: number;
  readonly params: GmmParams;
  readonly responsibilities: Mat | null;
  readonly logLikelihood: number | null;
}

function buildKMeansFrames(initialMeans: Mat): KMeansFrame[] {
  const result = kmeansFit(DATA, initialMeans, ROUNDS);
  const frames: KMeansFrame[] = [{ kind: 'init', round: 0, means: result.meansHistory[0]!, assignments: null }];
  for (let i = 0; i < ROUNDS; i++) {
    const assignments = result.assignmentsHistory[i]!;
    frames.push({ kind: 'E', round: i + 1, means: result.meansHistory[i]!, assignments });
    frames.push({ kind: 'M', round: i + 1, means: result.meansHistory[i + 1]!, assignments });
  }
  return frames;
}

function buildEmFrames(initialParams: GmmParams): EmFrame[] {
  const frames: EmFrame[] = [{ kind: 'init', round: 0, params: initialParams, responsibilities: null, logLikelihood: null }];
  let params = initialParams;
  for (let i = 0; i < ROUNDS; i++) {
    const step = gmmEmStep(DATA, params);
    frames.push({ kind: 'E', round: i + 1, params, responsibilities: step.responsibilities, logLikelihood: step.logLikelihood });
    frames.push({ kind: 'M', round: i + 1, params: step.params, responsibilities: step.responsibilities, logLikelihood: step.logLikelihood });
    params = step.params;
  }
  return frames;
}

export default function KMeansVsEM() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const { kMeansFrames, emFrames } = useMemo(() => {
    const rng = pcg32(SEED);
    const initialMeans = kmeansInit(rng, DATA, K);
    const initialParams: GmmParams = {
      components: initialMeans.map((mean) => ({ weight: 1 / K, mean, cov: ISOTROPIC_COV })),
    };
    return { kMeansFrames: buildKMeansFrames(initialMeans), emFrames: buildEmFrames(initialParams) };
  }, []);

  const stepCount = kMeansFrames.length;
  const clampedStep = Math.min(step, stepCount - 1);
  const kFrame = kMeansFrames[clampedStep]!;
  const emFrame = emFrames[clampedStep]!;

  const voronoi = useMemo(
    () => evalGrid(GRID_AXIS, GRID_AXIS, (x, y) => kmeansAssign([[x, y]], kFrame.means)[0]!),
    [kFrame.means],
  );
  const colorAt = useMemo(() => categoricalScale(tokens.series), [tokens.series]);

  const density = useMemo(
    () => evalGrid(GRID_AXIS, GRID_AXIS, (x, y) => gmmPdf([x, y], emFrame.params)),
    [emFrame.params],
  );
  const densityFill = useMemo(() => {
    let peak = 0;
    for (const row of density.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak]);
  }, [density]);

  const labels = kMeansFrames.map((f) => (f.kind === 'init' ? 'Initial state' : `${f.kind}-step ${f.round}`));

  return (
    <div>
      <div className="widget-grid">
        <Plot width={300} height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="K-means: hard assignment and Voronoi regions">
          <Heatmap data={voronoi} interpolator={colorAt} opacity={0.12} />
          <ContourField data={voronoi} levels={[0.5]} color={tokens.color.borderStrong} />
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
          <ScatterField
            points={DATA.map((p, i) => ({
              x: p[0]!,
              y: p[1]!,
              color: kFrame.assignments ? tokens.series[kFrame.assignments[i]!] : tokens.color.inkFaint,
              size: 3.5,
            }))}
          />
          <ScatterField
            points={kFrame.means.map((m, k) => ({ x: m[0]!, y: m[1]!, color: tokens.series[k], shape: 'cross', size: 9 }))}
          />
        </Plot>
        <Plot width={300} height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="EM: soft responsibility and mixture density">
          <ContourField data={density} levelCount={7} color={tokens.color.inkFaint} fill={densityFill} z={-2} opacity={0.5} />
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
          <ScatterField
            points={DATA.map((p, i) => ({
              x: p[0]!,
              y: p[1]!,
              color: emFrame.responsibilities ? blendColors(emFrame.responsibilities[i]!, tokens.series) : tokens.color.inkFaint,
              size: 3.5,
            }))}
          />
          <ScatterField
            points={emFrame.params.components.map((c, k) => ({ x: c.mean[0]!, y: c.mean[1]!, color: tokens.series[k], shape: 'cross', size: 9 }))}
          />
        </Plot>
      </div>
      <StepThrough step={clampedStep} stepCount={stepCount} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        {emFrame.logLikelihood === null
          ? 'Both panels start from the same two means.'
          : `Incomplete-data log-likelihood at this point: ${emFrame.logLikelihood.toFixed(3)}. A point near the boundary stays near-white on the right; K-means forces it to one colour on the left.`}
      </p>
    </div>
  );
}
