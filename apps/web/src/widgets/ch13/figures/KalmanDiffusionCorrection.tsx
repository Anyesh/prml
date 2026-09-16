import { linspace, mvnPdf } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';

const PREV_MEAN = 0;
const PREV_VAR = 0.4;
const A = 1.15;
const GAMMA = 0.35;
const PREDICTED_MEAN = A * PREV_MEAN + 0.9;
const PREDICTED_VAR = A * A * PREV_VAR + GAMMA;
const C = 0.7;
const SIGMA = 0.3;
const OBSERVATION = 2.2;

const ZS = linspace(-2, 5, 200);

function gaussianCurve(mean: number, variance: number) {
  return ZS.map((z) => [z, mvnPdf([z], { mean: [mean], cov: [[variance]] })] as const);
}

function emissionCurve() {
  return ZS.map((z) => [z, mvnPdf([OBSERVATION], { mean: [C * z], cov: [[SIGMA]] })] as const);
}

export default function KalmanDiffusionCorrection() {
  const tokens = useResolvedTokens();
  const prev = gaussianCurve(PREV_MEAN, PREV_VAR);
  const predicted = gaussianCurve(PREDICTED_MEAN, PREDICTED_VAR);
  const emission = emissionCurve();

  const innovationVar = 1 / (1 / PREDICTED_VAR + (C * C) / SIGMA);
  const innovationMean =
    innovationVar * (PREDICTED_MEAN / PREDICTED_VAR + (C * OBSERVATION) / SIGMA);
  const corrected = gaussianCurve(innovationMean, innovationVar);

  return (
    <Plot width={360} height={220} xDomain={[-2, 5]} yDomain={[0, 1.1]} label="Diffusion widens the prediction; the observation narrows it back">
      <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
      <Curve points={prev} color={tokens.color.inkFaint} width={1.5} dash="dashed" />
      <Curve points={predicted} color={tokens.series[1]!} width={2} />
      <Curve points={emission} color={tokens.series[2]!} width={1.5} dash="dotted" />
      <Curve points={corrected} color={tokens.color.accent} width={2.5} />
    </Plot>
  );
}
