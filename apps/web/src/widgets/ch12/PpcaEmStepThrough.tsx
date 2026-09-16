import { useMemo, useState } from 'react';
import { dataMean, mvnCovarianceEllipse, pcg32, ppcaFitEM, ppcaMarginalCov, standardNormal, type Ellipse, type Mat, type PpcaParams } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, VectorField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { ppcaDemoData } from './data.js';

import '../widgets.css';

export const title = 'EM for probabilistic PCA';
export const caption =
  'Step through E and M rounds. The latent axis swings toward the data direction and the ellipse tightens around it, while the log-likelihood only ever rises.';
export const figure = '12.2';

const DATA = ppcaDemoData();
const DOMAIN: readonly [number, number] = [-4, 4];
const MAX_ROUNDS = 8;
const INIT_SEED = 20260920;

function ellipsePoints(ellipse: Ellipse, samples = 64): (readonly [number, number])[] {
  const cosA = Math.cos(ellipse.angle);
  const sinA = Math.sin(ellipse.angle);
  return Array.from({ length: samples + 1 }, (_, i) => {
    const t = (i / samples) * 2 * Math.PI;
    const x = ellipse.rx * Math.cos(t);
    const y = ellipse.ry * Math.sin(t);
    return [ellipse.cx + x * cosA - y * sinA, ellipse.cy + x * sinA + y * cosA] as const;
  });
}

function buildInitial(data: Mat): PpcaParams {
  const rng = pcg32(INIT_SEED);
  const mean = dataMean(data);
  return { mean, w: [[0.3 * standardNormal(rng)], [0.3 * standardNormal(rng)]], sigma2: 1 };
}

export default function PpcaEmStepThrough() {
  const tokens = useResolvedTokens();
  const [step, setStep] = useState(0);

  const fit = useMemo(() => ppcaFitEM(DATA, buildInitial(DATA), MAX_ROUNDS), []);
  const clampedStep = Math.min(step, fit.paramsHistory.length - 1);
  const params = fit.paramsHistory[clampedStep]!;
  const ellipse = mvnCovarianceEllipse({ mean: params.mean, cov: ppcaMarginalCov(params.w, params.sigma2) }, 0.95);
  const latentDir = [params.w[0]![0]!, params.w[1]![0]!] as const;
  const latentNorm = Math.hypot(latentDir[0], latentDir[1]) || 1;

  const llPoints = fit.logLikelihoodHistory.map((v, i) => [i, v] as const);
  const labels = fit.paramsHistory.map((_, i) => (i === 0 ? 'Initial' : `Round ${i}`));
  const readoutIndex = Math.min(clampedStep, llPoints.length - 1);

  return (
    <div>
      <div className="widget-grid">
        <Plot height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Data with the current marginal covariance ellipse and latent axis">
          <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
          <ScatterField points={DATA.map((p, i) => ({ x: p[0]!, y: p[1]!, id: i }))} color={tokens.color.inkMuted} size={3} opacity={0.75} />
          <Curve points={ellipsePoints(ellipse)} color={tokens.color.accent} width={2} />
          <VectorField
            origins={[[params.mean[0]!, params.mean[1]!]]}
            field={() => [(2.2 * latentDir[0]) / latentNorm, (2.2 * latentDir[1]) / latentNorm]}
            color={tokens.series[0]!}
            maxLength={90}
          />
        </Plot>
        <Plot
          height={300}
          xDomain={[0, MAX_ROUNDS]}
          yDomain={[fit.logLikelihoodHistory[0]! - 5, fit.logLikelihoodHistory[fit.logLikelihoodHistory.length - 1]! + 5]}
          label="Marginal log-likelihood across EM rounds"
        >
          <Axes x={{ label: 'round' }} y={{ label: 'log p(X)' }} grid />
          <Curve points={llPoints} color={tokens.color.accent} width={2} />
          <ScatterField points={[{ x: readoutIndex, y: llPoints[readoutIndex]?.[1] ?? 0, color: tokens.color.accent, size: 4 }]} />
        </Plot>
      </div>
      <StepThrough step={clampedStep} stepCount={fit.paramsHistory.length} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        sigma² = {params.sigma2.toFixed(3)}.{' '}
        {clampedStep === 0
          ? 'A near-isotropic start: the ellipse is almost a circle.'
          : 'Each round narrows sigma² and turns the axis toward the data, never worsening the fit.'}
      </p>
    </div>
  );
}
