import { useState } from 'react';
import { gmmLogPdf, type GmmParams } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../../widgets.css';

/** One data point a shrinking component collapses onto, PRML's account of the eq. 9.15 singularity. */
const COLLAPSE_POINT: readonly [number, number] = [1.2, -0.6];
const BROAD_COV = [
  [3, 0],
  [0, 3],
];
const LOG_EPSILON_MIN = -6;
const LOG_EPSILON_MAX = 0;
const CURVE_SAMPLES = 60;

function paramsAt(epsilon: number): GmmParams {
  return {
    components: [
      { weight: 0.95, mean: [0, 0], cov: BROAD_COV },
      {
        weight: 0.05,
        mean: [...COLLAPSE_POINT],
        cov: [
          [epsilon, 0],
          [0, epsilon],
        ],
      },
    ],
  };
}

export default function CovarianceSingularity() {
  const [logEpsilon, setLogEpsilon] = useState(-1);
  const tokens = useResolvedTokens();
  const epsilon = 10 ** logEpsilon;
  const logDensityAtPoint = gmmLogPdf(COLLAPSE_POINT, paramsAt(epsilon));

  const curvePoints = Array.from({ length: CURVE_SAMPLES }, (_, i) => {
    const le = LOG_EPSILON_MIN + ((LOG_EPSILON_MAX - LOG_EPSILON_MIN) * i) / (CURVE_SAMPLES - 1);
    return [le, gmmLogPdf(COLLAPSE_POINT, paramsAt(10 ** le))] as const;
  });
  const maxY = Math.max(...curvePoints.map(([, y]) => y));
  const minY = Math.min(...curvePoints.map(([, y]) => y));

  return (
    <div>
      <Plot height={220} xDomain={[LOG_EPSILON_MIN, LOG_EPSILON_MAX]} yDomain={[minY, maxY + 1]} label="Log density at the collapsing point as its component's variance shrinks">
        <Axes x={{ label: 'log10(epsilon)' }} y={{ label: 'log p(x*)' }} grid />
        <Curve points={curvePoints} color={tokens.color.danger} width={2} />
      </Plot>
      <Slider
        label="Collapsing component's variance (epsilon)"
        value={logEpsilon}
        onChange={setLogEpsilon}
        min={LOG_EPSILON_MIN}
        max={LOG_EPSILON_MAX}
        step={0.05}
        format={(v) => (10 ** v).toExponential(1)}
        hint="Drag towards the left. Nothing about the model changes except one component's mean sitting on a single point."
      />
      <p className="widget-readout">
        At epsilon = {epsilon.toExponential(2)}, log p(x*) = {logDensityAtPoint.toFixed(2)}. It has no ceiling: this is the
        genuine singularity PRML 9.15 describes, not a numerical bug.
      </p>
    </div>
  );
}
