import { useMemo, useState } from 'react';
import { emKlGap, emLowerBound, gmmEStep, gmmLogLikelihood, linspace, pcg32, standardNormal, type GmmParams, type Mat } from '@prml/math';
import { Annotation, Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../widgets.css';

export const title = 'The lower bound, stepping up the curve';
export const caption =
  'Step forward. The E-step redraws the bound to touch the log-likelihood at the current point; the M-step slides along that bound to its peak, which is always at least as high.';
export const figure = '9.14';

const SEED = 20260916;
const CLUSTER_A_MEAN = -2;
const CLUSTER_B_MEAN = 2;
const CLUSTER_SIZE = 6;
const FIXED_MEAN = -2.2;
const COV: number[][] = [[1]];
const THETA_START = 0.6;
const THETA_MIN = -4;
const THETA_MAX = 4;
const GRID_SAMPLES = 90;
const ROUNDS = 3;

const DATA: Mat = (() => {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  for (let i = 0; i < CLUSTER_SIZE; i++) points.push([CLUSTER_A_MEAN + 0.6 * standardNormal(rng)]);
  for (let i = 0; i < CLUSTER_SIZE; i++) points.push([CLUSTER_B_MEAN + 0.6 * standardNormal(rng)]);
  return points;
})();

const THETA_GRID = linspace(THETA_MIN, THETA_MAX, GRID_SAMPLES);

function paramsAt(theta: number): GmmParams {
  return {
    components: [
      { weight: 0.5, mean: [FIXED_MEAN], cov: COV },
      { weight: 0.5, mean: [theta], cov: COV },
    ],
  };
}

function argmaxOverGrid(values: readonly number[], grid: readonly number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i]! > values[best]!) best = i;
  return grid[best]!;
}

const LL_CURVE = THETA_GRID.map((theta) => gmmLogLikelihood(DATA, paramsAt(theta)));

interface Frame {
  readonly kind: 'init' | 'E' | 'M';
  readonly round: number;
  readonly theta: number;
  readonly responsibilities: Mat | null;
}

const FRAMES: Frame[] = (() => {
  const frames: Frame[] = [{ kind: 'init', round: 0, theta: THETA_START, responsibilities: null }];
  let theta = THETA_START;
  for (let r = 0; r < ROUNDS; r++) {
    const responsibilities = gmmEStep(DATA, paramsAt(theta));
    frames.push({ kind: 'E', round: r + 1, theta, responsibilities });
    const boundCurve = THETA_GRID.map((t) => emLowerBound(DATA, responsibilities, paramsAt(t)));
    theta = argmaxOverGrid(boundCurve, THETA_GRID);
    frames.push({ kind: 'M', round: r + 1, theta, responsibilities });
  }
  return frames;
})();

export default function LowerBoundClimb() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const frame = FRAMES[step]!;
  const boundCurve = useMemo(
    () => (frame.responsibilities ? THETA_GRID.map((t) => [t, emLowerBound(DATA, frame.responsibilities!, paramsAt(t))] as const) : null),
    [frame.responsibilities],
  );
  const llPoints = THETA_GRID.map((t, i) => [t, LL_CURVE[i]!] as const);
  const currentLl = gmmLogLikelihood(DATA, paramsAt(frame.theta));
  const gap = frame.responsibilities ? emKlGap(DATA, frame.responsibilities, paramsAt(frame.theta)) : 0;

  const minY = Math.min(...LL_CURVE);
  const maxY = Math.max(...LL_CURVE);
  const pad = (maxY - minY) * 0.15;

  const labels = FRAMES.map((f) => (f.kind === 'init' ? 'Starting point' : `${f.kind}-step ${f.round}`));

  return (
    <div>
      <Plot height={300} xDomain={[THETA_MIN, THETA_MAX]} yDomain={[minY - pad, maxY + pad]} label="Log-likelihood curve against theta, with the current lower bound and position">
        <Axes x={{ label: 'theta (second component mean)' }} y={{ label: 'ln p(X|theta)' }} grid />
        <Curve points={llPoints} color={tokens.color.danger} width={2} />
        {boundCurve ? <Curve points={boundCurve} color={tokens.color.accent} width={2} dash="dashed" /> : null}
        <ScatterField points={[{ x: frame.theta, y: currentLl, color: tokens.color.ink, size: 5.5 }]} />
        <Legend
          entries={[
            { label: 'ln p(X|theta)', color: tokens.color.danger, mark: 'line' },
            { label: 'L(q,theta)', color: tokens.color.accent, mark: 'dashed-line' },
          ]}
        />
        {frame.kind !== 'init' ? <Annotation x={frame.theta} y={currentLl} text={`gap=${gap.toFixed(3)}`} dy={-16} /> : null}
      </Plot>
      <StepThrough step={step} stepCount={FRAMES.length} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        {frame.kind === 'init'
          ? `Starting at theta = ${frame.theta.toFixed(2)}, away from either cluster mean.`
          : frame.kind === 'E'
            ? `E-step: the bound now touches the curve exactly at theta = ${frame.theta.toFixed(2)} (gap = ${gap.toFixed(4)}).`
            : `M-step: theta moves to ${frame.theta.toFixed(2)}, the peak of the last bound. ln p(X|theta) = ${currentLl.toFixed(3)}.`}
      </p>
    </div>
  );
}
