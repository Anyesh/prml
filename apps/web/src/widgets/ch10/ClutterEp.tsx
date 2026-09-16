import { useMemo, useState } from 'react';
import {
  clutterMomentMatch,
  isotropicCavity,
  linspace,
  normalPdf,
  pcg32,
  refineIsotropicSite,
  standardNormal,
  type ClutterModel,
  type IsotropicGaussian,
  type IsotropicSite,
} from '@prml/math';
import { Axes, Bars, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../widgets.css';

export const title = 'Expectation propagation on the clutter problem';
export const caption =
  'Step through one sweep, point by point. Watch the posterior curve tighten around the true theta while the outliers barely move it, because each moment match down-weights whichever point looks like clutter.';
export const figure = '10.16';

const SEED = 20260917;
const TRUE_THETA = 2.5;
const MODEL: ClutterModel = { clutterWeight: 0.4, clutterVariance: 9 };
const PRIOR: IsotropicGaussian = { mean: [0], variance: 25 };
const UNINFORMATIVE_VARIANCE = 1e12;
const DOMAIN: readonly [number, number] = [-8, 8];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 100);

function syntheticData(): number[] {
  const rng = pcg32(SEED);
  const points: number[] = [];
  for (let i = 0; i < 5; i++) points.push(TRUE_THETA + standardNormal(rng));
  for (let i = 0; i < 3; i++) points.push(6 * standardNormal(rng));
  return points;
}

const DATA = syntheticData();

interface PointStep {
  readonly posterior: IsotropicGaussian;
  readonly signalProbability: number | null;
}

/**
 * Builds a per-point trajectory of one EP sweep by calling the cavity, moment-match, and
 * refine primitives in sequence, rather than only the sweep-level result `clutterEpSweep`
 * returns: a step-through widget needs the state after every point, not only after all of
 * them.
 */
function buildPointTrajectory(): PointStep[] {
  const sites: IsotropicSite[] = DATA.map(() => ({ logScale: 0, mean: [0], variance: UNINFORMATIVE_VARIANCE }));
  let posterior = PRIOR;
  const steps: PointStep[] = [{ posterior, signalProbability: null }];
  for (let n = 0; n < DATA.length; n++) {
    const cavity = isotropicCavity(posterior, sites[n]!);
    const moments = clutterMomentMatch([DATA[n]!], cavity, MODEL);
    sites[n] = refineIsotropicSite(cavity, moments);
    posterior = moments.posterior;
    steps.push({ posterior, signalProbability: moments.signalProbability });
  }
  return steps;
}

export default function ClutterEp() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const trajectory = useMemo(buildPointTrajectory, []);
  const clampedStep = Math.min(step, trajectory.length - 1);
  const current = trajectory[clampedStep]!;

  const posteriorCurve = GRID.map(
    (x) => [x, normalPdf(x, { mu: current.posterior.mean[0]!, sigma2: current.posterior.variance })] as const,
  );

  const bars = DATA.map((x, i) => {
    const seenYet = i < clampedStep;
    const isCurrent = i === clampedStep - 1;
    const p = isCurrent ? current.signalProbability! : seenYet ? trajectory[i + 1]!.signalProbability! : 0;
    return { at: i, value: seenYet ? p : 0, color: seenYet && p < 0.5 ? tokens.color.danger : tokens.color.accent };
  });

  return (
    <div>
      <Plot height={220} xDomain={DOMAIN} yDomain={[0, 0.45]} label="EP's current isotropic posterior over theta">
        <Axes x={{ label: 'theta' }} y={{ label: 'q(theta)' }} grid />
        <Curve points={posteriorCurve} color={tokens.series[0]!} width={2} />
        <ScatterField
          points={DATA.map((x, i) => ({
            x,
            y: 0.01,
            color: i < clampedStep ? tokens.color.ink : tokens.color.inkFaint,
            size: i === clampedStep - 1 ? 6 : 4,
            shape: 'cross' as const,
          }))}
        />
      </Plot>
      <Plot height={140} xDomain={[-0.5, DATA.length - 0.5]} yDomain={[0, 1]} label="Posterior probability each processed point is signal rather than clutter">
        <Axes x={{ label: 'point index' }} y={{ label: 'P(signal)' }} grid />
        <Bars bars={bars} thickness={0.5} />
      </Plot>
      <StepThrough
        step={clampedStep}
        stepCount={trajectory.length}
        onStep={setStep}
        labels={trajectory.map((_, i) => (i === 0 ? 'prior' : `after point ${i}`))}
      />
      <p className="widget-readout">
        Posterior mean {current.posterior.mean[0]!.toFixed(2)}, variance {current.posterior.variance.toFixed(2)},
        against a true theta of {TRUE_THETA}. A bar below 0.5 marks a point the moment match at that step judged more
        likely clutter than signal.
      </p>
    </div>
  );
}
