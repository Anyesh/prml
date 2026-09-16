import { pcg32, standardNormal, variationalLogisticFit } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
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

export default function XiConvergence() {
  const tokens = useResolvedTokens();
  const xiInit = new Array(DESIGN.length).fill(1);
  const fit = variationalLogisticFit(DESIGN, labels, PRIOR, xiInit, ROUNDS);

  const seriesPerPoint = DESIGN.map((_, n) => fit.xiHistory.map((xi, round) => [round, xi[n]!] as const));

  return (
    <Plot height={200} xDomain={[0, ROUNDS - 1]} yDomain={[0, Math.max(...fit.xiHistory[fit.xiHistory.length - 1]!) * 1.15]} label="Every observation's variational parameter xi_n across rounds">
      <Axes x={{ label: 'round' }} y={{ label: 'xi_n' }} grid />
      {seriesPerPoint.map((pts, i) => (
        <Curve key={i} points={pts} color={tokens.color.inkFaint} width={1} />
      ))}
    </Plot>
  );
}
