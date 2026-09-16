import { useState } from 'react';
import {
  isotropicPrior,
  mvnCovarianceEllipse,
  pcg32,
  polynomialBasis,
  standardNormal,
  updatePosterior,
  type Ellipse,
  type WeightPosterior,
} from '@prml/math';
import { Axes, CovarianceEllipse, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../../widgets.css';

const TRUE_W = [-0.3, 0.5] as const;
const ALPHA = 2;
const BETA = 25;
const NOISE_STD = 0.2;
const SEED = 20260402;
const OBS_COUNT = 8;
const PHI = polynomialBasis(1);
const DOMAIN: readonly [number, number] = [-2, 2];
const MASS = 0.95;

interface Observation {
  readonly x: number;
  readonly t: number;
}

function generateObservations(): Observation[] {
  const rng = pcg32(SEED);
  return Array.from({ length: OBS_COUNT }, () => {
    const x = rng.next() * 2 - 1;
    return { x, t: TRUE_W[0] + TRUE_W[1] * x + NOISE_STD * standardNormal(rng) };
  });
}

const OBSERVATIONS = generateObservations();

/** Prior followed by the posterior after each successive observation, index 0 = prior. */
const POSTERIORS: WeightPosterior[] = (() => {
  let posterior = isotropicPrior(2, ALPHA);
  const list = [posterior];
  for (const o of OBSERVATIONS) {
    posterior = updatePosterior(posterior, PHI(o.x), o.t, BETA);
    list.push(posterior);
  }
  return list;
})();

function ellipseArea(ellipse: Ellipse): number {
  return Math.PI * ellipse.rx * ellipse.ry;
}

const ELLIPSES = POSTERIORS.map((p) => mvnCovarianceEllipse(p, MASS));
const STEP_LABELS = ['Prior', ...OBSERVATIONS.map((_, i) => `${i + 1} observation${i === 0 ? '' : 's'}`)];

export default function PrecisionAccumulates() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const visible = POSTERIORS.slice(0, step + 1);
  const firstArea = ellipseArea(ELLIPSES[0]!);
  const currentArea = ellipseArea(ELLIPSES[step]!);

  return (
    <div>
      <Plot
        height={260}
        xDomain={DOMAIN}
        yDomain={DOMAIN}
        equalAspect
        label="95% covariance ellipse of the posterior after each observation, all overlaid"
      >
        <Axes x={{ label: 'w₀' }} y={{ label: 'w₁' }} grid zeroLine />
        {visible.map((posterior, i) => (
          <CovarianceEllipse
            key={i}
            mean={posterior.mean}
            cov={posterior.cov}
            levels={[MASS]}
            color={i === step ? tokens.color.accent : tokens.color.inkFaint}
            width={i === step ? 2.25 : 1}
            opacity={i === step ? 1 : 0.25 + 0.55 * (i / (ELLIPSES.length - 1))}
          />
        ))}
        <ScatterField
          points={[{ x: TRUE_W[0], y: TRUE_W[1], shape: 'cross', color: tokens.color.danger, size: 7 }]}
          label={() => 'True weights'}
        />
      </Plot>
      <StepThrough step={step} stepCount={ELLIPSES.length} onStep={setStep} labels={STEP_LABELS} />
      <p className="widget-readout">
        {step === 0
          ? `Before any data, the 95% ellipse covers an area of ${firstArea.toFixed(2)}.`
          : `After ${step} observation${step === 1 ? '' : 's'}, the 95% ellipse covers ${currentArea.toFixed(3)}, down from ${firstArea.toFixed(2)} at the prior. Step through and watch it shrink at every single step, never once widen.`}
      </p>
    </div>
  );
}
