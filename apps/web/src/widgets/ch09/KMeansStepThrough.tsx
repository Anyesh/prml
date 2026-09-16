import { useMemo, useState } from 'react';
import { evalGrid, kmeansAssign, kmeansDistortion, kmeansFit, kmeansInit, linspace, pcg32, type Mat } from '@prml/math';
import { Axes, ContourField, Curve, Heatmap, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { kmeansDemoData } from './data.js';
import { categoricalInterpolator } from './colors.js';
import '../widgets.css';

export const title = 'Stepping K-means';
export const caption =
  'Press through E-steps and M-steps one at a time. Watch the Voronoi regions redraw at each M-step and the distortion plot fall at every single step, never once rise.';
export const figure = '9.1';

const DATA = kmeansDemoData();
const K = 3;
const MAX_ROUNDS = 7;
const BASE_SEED = 20260912;
const DOMAIN: readonly [number, number] = [-5, 5];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 56);

interface Frame {
  readonly kind: 'init' | 'E' | 'M';
  readonly round: number;
  readonly means: Mat;
  readonly assignments: readonly number[] | null;
  readonly distortion: number | null;
}

function buildFrames(data: Mat, initialMeans: Mat, maxRounds: number): Frame[] {
  const result = kmeansFit(data, initialMeans, maxRounds);
  const frames: Frame[] = [{ kind: 'init', round: 0, means: result.meansHistory[0]!, assignments: null, distortion: null }];
  for (let i = 0; i < maxRounds; i++) {
    const assignments = result.assignmentsHistory[i]!;
    const meansAfterM = result.meansHistory[i + 1]!;
    frames.push({ kind: 'E', round: i + 1, means: result.meansHistory[i]!, assignments, distortion: result.distortionHistory[i]! });
    frames.push({ kind: 'M', round: i + 1, means: meansAfterM, assignments, distortion: kmeansDistortion(data, meansAfterM, assignments) });
  }
  return frames;
}

export default function KMeansStepThrough() {
  const [seedTick, setSeedTick] = useState(0);
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const frames = useMemo(() => {
    const rng = pcg32(BASE_SEED, seedTick + 1);
    const initialMeans = kmeansInit(rng, DATA, K);
    return buildFrames(DATA, initialMeans, MAX_ROUNDS);
  }, [seedTick]);

  const clampedStep = Math.min(step, frames.length - 1);
  const frame = frames[clampedStep]!;

  const grid = useMemo(
    () => evalGrid(GRID_AXIS, GRID_AXIS, (x, y) => kmeansAssign([[x, y]], frame.means)[0]!),
    [frame.means],
  );
  const colorAt = useMemo(() => categoricalInterpolator(tokens.series), [tokens.series]);

  const maxDistortion = Math.max(...frames.map((f) => f.distortion ?? 0));
  const distortionPoints = frames
    .map((f, i) => [i, f.distortion] as const)
    .filter((p): p is readonly [number, number] => p[1] !== null);

  const labels = frames.map((f) => (f.kind === 'init' ? 'Initial means' : `${f.kind}-step ${f.round}`));

  return (
    <div>
      <div className="widget-grid">
        <Plot
          width={300}
          height={300}
          xDomain={DOMAIN}
          yDomain={DOMAIN}
          equalAspect
          label="Data coloured by current cluster assignment, with Voronoi regions for the current means"
        >
          <Heatmap data={grid} interpolator={colorAt} opacity={0.14} />
          <ContourField data={grid} levels={[0.5, 1.5]} color={tokens.color.borderStrong} />
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
          <ScatterField
            points={DATA.map((p, i) => ({
              x: p[0]!,
              y: p[1]!,
              color: frame.assignments ? tokens.series[frame.assignments[i]!] : tokens.color.inkFaint,
              size: 3.5,
            }))}
            label={(_, i) => `Point ${i + 1}`}
          />
          <ScatterField
            points={frame.means.map((m, k) => ({ x: m[0]!, y: m[1]!, color: tokens.series[k], shape: 'cross', size: 9 }))}
            label={(_, k) => `Mean ${k + 1}`}
          />
        </Plot>
        <Plot height={300} xDomain={[0, frames.length - 1]} yDomain={[0, maxDistortion * 1.05]} label="Distortion J across steps">
          <Axes x={{ label: 'step' }} y={{ label: 'J' }} grid />
          <Curve points={distortionPoints} color={tokens.color.accent} width={2} />
          <ScatterField
            points={[{ x: clampedStep, y: frame.distortion ?? 0 }]}
            color={tokens.color.accent}
            size={4}
          />
        </Plot>
      </div>
      <StepThrough step={clampedStep} stepCount={frames.length} onStep={setStep} labels={labels} />
      <button
        type="button"
        className="widget-reset"
        onClick={() => {
          setSeedTick((t) => t + 1);
          setStep(0);
        }}
      >
        New random start
      </button>
      <p className="widget-readout">
        {frame.kind === 'init'
          ? 'Three means dropped at random into the data. Step forward to assign points.'
          : frame.kind === 'E'
            ? `E-step ${frame.round}: every point joins its nearest mean. J = ${frame.distortion!.toFixed(3)}.`
            : `M-step ${frame.round}: each mean moves to the average of its points. J = ${frame.distortion!.toFixed(3)}.`}
      </p>
    </div>
  );
}
