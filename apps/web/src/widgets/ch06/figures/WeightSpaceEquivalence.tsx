import {
  dot,
  gaussianBasis,
  gpPriorSample,
  kernelFromFeatureMap,
  linspace,
  pcg32,
  scaleKernel,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const ALPHA = 2;
const CENTRES = linspace(0, 1, 8);
const SCALE = 0.12;
const PHI = gaussianBasis(CENTRES, SCALE, { bias: true });
const KERNEL = scaleKernel(kernelFromFeatureMap((v: readonly number[]) => PHI(v[0]!)), 1 / ALPHA);
const GRID = linspace(0, 1, 100);
const GRID_VECS = GRID.map((x) => [x]);
const SAMPLE_COUNT = 5;

const rng = pcg32(20260916);
const WEIGHT_SPACE_CURVES = Array.from({ length: SAMPLE_COUNT }, () => {
  const w = PHI(0).map(() => standardNormal(rng) / Math.sqrt(ALPHA));
  return GRID.map((x) => dot(PHI(x), w));
});

const FUNCTION_SPACE_CURVES = Array.from({ length: SAMPLE_COUNT }, (_, i) => gpPriorSample(rng.fork(100 + i), KERNEL, GRID_VECS));

const X1 = 0.2;
const X2 = 0.7;
const WEIGHT_SPACE_COV = dot(PHI(X1), PHI(X2)) / ALPHA;
const KERNEL_COV = KERNEL([X1], [X2]);

export default function WeightSpaceEquivalence() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[0, 1]} yDomain={[-3, 3]} label="Functions sampled by drawing weights from the prior and evaluating phi(x)^T w">
        <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid zeroLine />
        {WEIGHT_SPACE_CURVES.map((curve, i) => (
          <Curve key={i} points={GRID.map((x, j) => [x, curve[j]!] as const)} color={tokens.series[0]!} width={1.25} />
        ))}
      </Plot>
      <Plot height={200} xDomain={[0, 1]} yDomain={[-3, 3]} label="Functions sampled directly from the Gaussian process this weight prior induces">
        <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid zeroLine />
        {FUNCTION_SPACE_CURVES.map((curve, i) => (
          <Curve key={i} points={GRID.map((x, j) => [x, curve[j]!] as const)} color={tokens.series[1]!} width={1.25} />
        ))}
      </Plot>
      <p className="widget-readout">
        {`Same model, two generators. cov[y(${X1}), y(${X2})] from the weight-space formula (6.54) is ${WEIGHT_SPACE_COV.toFixed(6)}; the kernel built from the same basis gives ${KERNEL_COV.toFixed(6)} at the same pair, because they are the same formula.`}
      </p>
    </div>
  );
}
