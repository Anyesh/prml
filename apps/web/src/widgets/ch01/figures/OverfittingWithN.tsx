import {
  designMatrix,
  linspace,
  maximumLikelihoodWeights,
  meanSquaredError,
  pcg32,
  polynomialBasis,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const DEGREE = 9;
const BASIS = polynomialBasis(DEGREE);
const FIT_GRID = linspace(0, 1, 161);
const TEST = sampleSet(200, 99);

function sampleSet(n: number, streamId: number): { xs: number[]; ts: number[] } {
  const rng = pcg32(SEED, streamId);
  const xs = Array.from({ length: n }, (_, i) => i / (n - 1));
  const ts = xs.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
  return { xs, ts };
}

function fitAt(n: number, streamId: number) {
  const set = sampleSet(n, streamId);
  const design = designMatrix(set.xs, BASIS);
  const weights = maximumLikelihoodWeights(design, set.ts);
  const curve = FIT_GRID.map((x) => [x, BASIS(x).reduce((s, v, i) => s + v * weights[i]!, 0)] as const);
  const testRms = Math.sqrt(meanSquaredError(designMatrix(TEST.xs, BASIS), TEST.ts, weights));
  return { set, curve, testRms };
}

const SMALL = fitAt(15, 1);
const LARGE = fitAt(100, 1);

export default function OverfittingWithN() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label={`Degree-9 fit at N = ${SMALL.set.xs.length}`}>
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={SMALL.curve} color={tokens.series[0]!} width={2} />
        <ScatterField points={SMALL.set.xs.map((x, i) => ({ x, y: SMALL.set.ts[i]!, id: i }))} color={tokens.color.ink} size={3} />
      </Plot>
      <Plot height={220} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label={`Degree-9 fit at N = ${LARGE.set.xs.length}`}>
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={LARGE.curve} color={tokens.series[1]!} width={2} />
        <ScatterField points={LARGE.set.xs.map((x, i) => ({ x, y: LARGE.set.ts[i]!, id: i }))} color={tokens.color.ink} size={2.5} />
      </Plot>
      <p className="widget-readout">
        {`Same M = 9, same noise level. Test RMS falls from ${SMALL.testRms.toFixed(3)} at N = ${SMALL.set.xs.length} to ${LARGE.testRms.toFixed(3)} at N = ${LARGE.set.xs.length}: more data lets the same flexible model be trusted.`}
      </p>
    </div>
  );
}
