import {
  designMatrix,
  linspace,
  maximumLikelihoodWeights,
  meanSquaredError,
  pcg32,
  polynomialBasis,
  regularisedWeights,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N = 10;
const DEGREE = 9;
const BASIS = polynomialBasis(DEGREE);
const FIT_GRID = linspace(0, 1, 161);
const LN_LAMBDA_GRID = linspace(-30, 0, 61);

const rng = pcg32(SEED, 1);
const XS = Array.from({ length: N }, (_, i) => i / (N - 1));
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
const DESIGN = designMatrix(XS, BASIS);

const testRng = pcg32(SEED, 2);
const TEST_XS = Array.from({ length: 100 }, (_, i) => i / 99);
const TEST_TS = TEST_XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(testRng));
const TEST_DESIGN = designMatrix(TEST_XS, BASIS);

function weightsAt(lnLambda: number): number[] {
  return lnLambda <= -29 ? maximumLikelihoodWeights(DESIGN, TS) : regularisedWeights(DESIGN, TS, Math.exp(lnLambda));
}

function curveFor(lnLambda: number) {
  const w = weightsAt(lnLambda);
  return FIT_GRID.map((x) => [x, BASIS(x).reduce((s, v, i) => s + v * w[i]!, 0)] as const);
}

const RMS_CURVE = LN_LAMBDA_GRID.map((ln) => {
  const w = weightsAt(ln);
  return {
    ln,
    trainRms: Math.sqrt(meanSquaredError(DESIGN, TS, w)),
    testRms: Math.sqrt(meanSquaredError(TEST_DESIGN, TEST_TS, w)),
  };
});
const BEST = RMS_CURVE.reduce((best, r) => (r.testRms < best.testRms ? r : best));

export default function RegularisationSweep() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label="Degree-9 fit at two regularisation strengths">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={curveFor(-18)} color={tokens.series[0]!} width={2} />
        <Curve points={curveFor(0)} color={tokens.series[1]!} width={2} />
        <ScatterField points={XS.map((x, i) => ({ x, y: TS[i]!, id: i }))} color={tokens.color.ink} size={4} />
      </Plot>
      <Plot height={220} xDomain={[-30, 0]} yDomain={[0, 1]} label="RMS error against ln lambda">
        <Axes x={{ label: 'ln λ' }} y={{ label: 'ERMS' }} grid />
        <Curve points={RMS_CURVE.map((r) => [r.ln, r.trainRms] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={RMS_CURVE.map((r) => [r.ln, r.testRms] as const)} color={tokens.series[1]!} width={2} />
        <Rule x={BEST.ln} color={tokens.color.ink} label={`best: ln λ = ${BEST.ln.toFixed(0)}`} />
      </Plot>
      <p className="widget-readout">
        {`ln λ = -18 (blue): test RMS ${Math.sqrt(meanSquaredError(TEST_DESIGN, TEST_TS, weightsAt(-18))).toFixed(3)}. ln λ = 0 (orange): test RMS ${Math.sqrt(meanSquaredError(TEST_DESIGN, TEST_TS, weightsAt(0))).toFixed(3)}, over-regularised. The lowest test RMS on this sweep is ${BEST.testRms.toFixed(3)} at ln λ = ${BEST.ln.toFixed(0)}.`}
      </p>
    </div>
  );
}
