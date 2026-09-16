import { pcg32, standardNormal, updateLogisticAlpha, variationalLogisticFit } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const N_PER_CLASS = 14;
const PRIOR_COV = [[4, 0, 0], [0, 4, 0], [0, 0, 4]];
const ROUNDS = 8;
const ALPHA_ROUNDS = 6;

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

export default function AlphaEvolution() {
  const tokens = useResolvedTokens();
  let alpha = 1;
  const trace: (readonly [number, number])[] = [[0, alpha]];

  for (let round = 0; round < ALPHA_ROUNDS; round++) {
    const prior = { mean: [0, 0, 0], cov: PRIOR_COV.map((row) => row.map((v) => v / alpha)) };
    const xiInit = new Array(DESIGN.length).fill(1);
    const fit = variationalLogisticFit(DESIGN, labels, prior, xiInit, ROUNDS);
    const posterior = fit.posteriorHistory[fit.posteriorHistory.length - 1]!;
    alpha = updateLogisticAlpha(posterior, alpha);
    trace.push([round + 1, alpha]);
  }

  return (
    <Plot height={200} xDomain={[0, ALPHA_ROUNDS]} yDomain={[0, Math.max(...trace.map((p) => p[1])) * 1.15]} label="The re-estimated prior precision alpha across outer rounds">
      <Axes x={{ label: 'outer round' }} y={{ label: 'alpha' }} grid />
      <Curve points={trace} color={tokens.series[0]!} width={2} />
    </Plot>
  );
}
