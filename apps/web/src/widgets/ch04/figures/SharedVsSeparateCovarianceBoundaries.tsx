import {
  evalGrid,
  fitSeparateCovarianceGaussian,
  fitSharedCovarianceGaussian,
  linspace,
  mvnSample,
  pcg32,
  posteriorSeparateCovariance,
  posteriorSharedCovariance,
} from '@prml/math';
import { Axes, ContourField, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MEAN1 = [-1.5, 0];
const MEAN2 = [1.5, 0.5];
const COV1 = [
  [1.6, 0.9],
  [0.9, 0.6],
];
const COV2 = [
  [0.4, -0.1],
  [-0.1, 1.5],
];
const N = 50;
const DOMAIN: readonly [number, number] = [-5, 5];
const GRID = linspace(-5, 5, 100);

const rng = pcg32(20260421);
const CLASS1 = Array.from({ length: N }, () => mvnSample(rng, { mean: MEAN1, cov: COV1 }));
const CLASS2 = Array.from({ length: N }, () => mvnSample(rng, { mean: MEAN2, cov: COV2 }));

const SHARED_FIT = fitSharedCovarianceGaussian([CLASS1, CLASS2]);
const SEPARATE_FIT = fitSeparateCovarianceGaussian([CLASS1, CLASS2]);
const SHARED_FIELD = evalGrid(GRID, GRID, (x, y) => posteriorSharedCovariance([x, y], SHARED_FIT)[0]!);
const SEPARATE_FIELD = evalGrid(GRID, GRID, (x, y) => posteriorSeparateCovariance([x, y], SEPARATE_FIT)[0]!);

export default function SharedVsSeparateCovarianceBoundaries() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Shared-covariance (linear) versus separate-covariance (quadratic) boundary">
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      <ContourField data={SHARED_FIELD} levels={[0.5]} color={tokens.color.accent} lineWidth={2} />
      <ContourField data={SEPARATE_FIELD} levels={[0.5]} color={tokens.color.danger} lineWidth={2} />
      <ScatterField points={CLASS1.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `1-${i}` }))} color={tokens.series[0]} size={2.5} opacity={0.6} />
      <ScatterField points={CLASS2.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `2-${i}` }))} color={tokens.series[1]} size={2.5} opacity={0.6} />
      <Legend
        entries={[
          { label: 'shared Σ: linear', color: tokens.color.accent, mark: 'line' },
          { label: 'separate Σ: quadratic', color: tokens.color.danger, mark: 'line' },
        ]}
        placement="top-right"
      />
    </Plot>
  );
}
