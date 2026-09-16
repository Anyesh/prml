import { evalGrid, fitLaplaceLogisticPosterior, linspace, mvnLogPdf, pcg32, standardNormal } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(20260441);
const N_PER_CLASS = 15;
const XS0 = Array.from({ length: N_PER_CLASS }, () => -1.5 + standardNormal(rng));
const XS1 = Array.from({ length: N_PER_CLASS }, () => 1.5 + standardNormal(rng));
const DESIGN = [...XS0, ...XS1].map((x) => [1, x]);
const TARGETS = [...XS0.map(() => 0), ...XS1.map(() => 1)];

const POSTERIOR = fitLaplaceLogisticPosterior(DESIGN, TARGETS, {
  mean: [0, 0],
  covariance: [
    [9, 0],
    [0, 9],
  ],
});

const W0_GRID = linspace(POSTERIOR.mean[0]! - 4, POSTERIOR.mean[0]! + 4, 90);
const W1_GRID = linspace(POSTERIOR.mean[1]! - 4, POSTERIOR.mean[1]! + 4, 90);
const POSTERIOR_MVN = { mean: POSTERIOR.mean, cov: POSTERIOR.covariance };
const DENSITY = evalGrid(W0_GRID, W1_GRID, (w0, w1) => Math.exp(mvnLogPdf([w0, w1], POSTERIOR_MVN)));

export default function PosteriorContourVsMAP() {
  const tokens = useResolvedTokens();
  const peak = Math.max(...DENSITY.values.flat());

  return (
    <Plot
      height={260}
      xDomain={[W0_GRID[0]!, W0_GRID[W0_GRID.length - 1]!]}
      yDomain={[W1_GRID[0]!, W1_GRID[W1_GRID.length - 1]!]}
      equalAspect
      label="The Laplace posterior in weight space, mode plus Hessian"
    >
      <ContourField data={DENSITY} levelCount={8} color={tokens.color.inkFaint} fill={sequentialScale([0, peak])} z={-1} />
      <Axes x={{ label: 'w₀' }} y={{ label: 'w₁' }} grid zeroLine />
      <ScatterField points={[{ x: POSTERIOR.mean[0]!, y: POSTERIOR.mean[1]!, id: 'map', shape: 'cross', size: 8, color: tokens.color.danger }]} />
    </Plot>
  );
}
