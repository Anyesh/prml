import { pcg32, standardNormal, submatrix, variationalLogisticFit } from '@prml/math';
import { Axes, CovarianceEllipse, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N_PER_CLASS = 14;
const PRIOR = { mean: [0, 0, 0], cov: [[4, 0, 0], [0, 4, 0], [0, 0, 4]] };
const ROUNDS = 8;

function syntheticData() {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < N_PER_CLASS; i++) {
    points.push([-1.3 + 0.7 * standardNormal(rng), -1 + 0.7 * standardNormal(rng)]);
    labels.push(0);
  }
  for (let i = 0; i < N_PER_CLASS; i++) {
    points.push([1.3 + 0.7 * standardNormal(rng), 1 + 0.7 * standardNormal(rng)]);
    labels.push(1);
  }
  return { points, labels };
}

const { points, labels } = syntheticData();
const DESIGN = points.map((p) => [1, p[0]!, p[1]!]);

export default function PosteriorVsPriorEllipse() {
  const tokens = useResolvedTokens();
  const xiInit = new Array(DESIGN.length).fill(1);
  const fit = variationalLogisticFit(DESIGN, labels, PRIOR, xiInit, ROUNDS);
  const posterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;

  const priorMarginal = { mean: [PRIOR.mean[1]!, PRIOR.mean[2]!], cov: submatrix(PRIOR.cov, [1, 2], [1, 2]) };
  const posteriorMarginal = { mean: [posterior.mean[1]!, posterior.mean[2]!], cov: submatrix(posterior.cov, [1, 2], [1, 2]) };

  return (
    <Plot width={280} height={280} xDomain={[-4, 4]} yDomain={[-4, 4]} equalAspect label="The prior over (w1, w2) shrinking to the posterior after seeing the data">
      <Axes x={{ label: 'w1' }} y={{ label: 'w2' }} grid zeroLine />
      <CovarianceEllipse mean={priorMarginal.mean} cov={priorMarginal.cov} levels={[0.9]} color={tokens.color.inkFaint} width={1.5} dash="dashed" />
      <CovarianceEllipse mean={posteriorMarginal.mean} cov={posteriorMarginal.cov} levels={[0.9]} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
