import { useMemo, useState } from 'react';
import { pcaFitCov, principalCurveFit, projectToPolyline, type Mat } from '@prml/math';
import { Axes, Curve, divergingScale, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { spiralData } from './data.js';

import '../widgets.css';

export const title = 'Fitting a principal curve';
export const caption =
  'Step the curve through its self-consistency updates. Colour follows the position each point takes along the curve, a latent coordinate no straight line could assign correctly.';
export const figure = '12.19';

const { data: DATA } = spiralData();
const DOMAIN: readonly [number, number] = [-7, 7];
const NUM_VERTICES = 8;
const MAX_ROUNDS = 6;
const BANDWIDTH = 1.4;

function initialCurve(data: Mat): Mat {
  const pca = pcaFitCov(data);
  const projections = data.map((x) => {
    const centered = [x[0]! - pca.mean[0]!, x[1]! - pca.mean[1]!];
    return centered[0]! * pca.components[0]![0]! + centered[1]! * pca.components[0]![1]!;
  });
  const lo = Math.min(...projections);
  const hi = Math.max(...projections);
  return Array.from({ length: NUM_VERTICES }, (_, i) => {
    const t = lo + ((hi - lo) * i) / (NUM_VERTICES - 1);
    return [pca.mean[0]! + t * pca.components[0]![0]!, pca.mean[1]! + t * pca.components[0]![1]!];
  });
}

export default function PrincipalCurveStepThrough() {
  const tokens = useResolvedTokens();
  const [step, setStep] = useState(0);

  const fit = useMemo(() => principalCurveFit(DATA, initialCurve(DATA), MAX_ROUNDS, BANDWIDTH), []);
  const clampedStep = Math.min(step, fit.curveHistory.length - 1);
  const curve = fit.curveHistory[clampedStep]!;

  const arclengths = DATA.map((x) => projectToPolyline(x, curve).arclength);
  const lo = Math.min(...arclengths);
  const hi = Math.max(...arclengths);
  const colorAt = divergingScale([lo, hi], (lo + hi) / 2);

  const labels = fit.curveHistory.map((_, i) => (i === 0 ? 'Initial (linear PCA)' : `Round ${i}`));

  return (
    <div>
      <Plot height={340} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Data with the current fitted curve, points coloured by their position along it">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <ScatterField points={DATA.map((p, i) => ({ x: p[0]!, y: p[1]!, color: colorAt(arclengths[i]!), size: 3.5 }))} />
        <Curve points={curve.map((p) => [p[0]!, p[1]!] as const)} color={tokens.color.accent} width={2.5} />
      </Plot>
      <StepThrough step={clampedStep} stepCount={fit.curveHistory.length} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        {clampedStep === 0
          ? 'Round 0 is the straight line linear PCA would draw: a single global direction through data that has none.'
          : 'Each round moves every vertex to the average of the data nearest it in arclength, bending the line to fit the roll.'}
      </p>
    </div>
  );
}
