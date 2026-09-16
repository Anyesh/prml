import { emKlGap, gmmEStep, pcg32, standardNormal, type GmmParams, type Mat } from '@prml/math';
import { Annotation, Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260916;
const CLUSTER_A_MEAN = -2;
const CLUSTER_B_MEAN = 2;
const CLUSTER_SIZE = 6;
const FIXED_MEAN = -2.2;
const COV: number[][] = [[1]];
const THETA_OLD = 0.6;

const DATA: Mat = (() => {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  for (let i = 0; i < CLUSTER_SIZE; i++) points.push([CLUSTER_A_MEAN + 0.6 * standardNormal(rng)]);
  for (let i = 0; i < CLUSTER_SIZE; i++) points.push([CLUSTER_B_MEAN + 0.6 * standardNormal(rng)]);
  return points;
})();

const PARAMS: GmmParams = {
  components: [
    { weight: 0.5, mean: [FIXED_MEAN], cov: COV },
    { weight: 0.5, mean: [THETA_OLD], cov: COV },
  ],
};

const Q_EXACT = gmmEStep(DATA, PARAMS);
/** Pulled 60% of the way towards a uniform 0.5/0.5 split, so it is a genuinely different distribution from the E-step optimum. */
const Q_PERTURBED = Q_EXACT.map((row) => {
  const mixed = row.map((r) => 0.4 * r + 0.6 * 0.5);
  const total = mixed[0]! + mixed[1]!;
  return mixed.map((v) => v / total);
});

const GAP_EXACT = emKlGap(DATA, Q_EXACT, PARAMS);
const GAP_PERTURBED = emKlGap(DATA, Q_PERTURBED, PARAMS);

export default function GapComparison() {
  const tokens = useResolvedTokens();
  const maxY = Math.max(GAP_EXACT, GAP_PERTURBED) * 1.2 || 1;

  return (
    <Plot height={200} xDomain={[-0.5, 1.5]} yDomain={[0, maxY]} label="KL gap at the E-step optimum against a perturbed q, same theta">
      <Axes x={{ label: 'q', ticks: [0, 1], format: (v) => (v === 0 ? 'E-step q' : 'perturbed q') }} y={{ label: 'KL(q||p)' }} grid />
      <ScatterField
        points={[
          { x: 0, y: GAP_EXACT, color: tokens.color.success, size: 7 },
          { x: 1, y: GAP_PERTURBED, color: tokens.color.danger, size: 7 },
        ]}
      />
      <Annotation x={0} y={GAP_EXACT} text={GAP_EXACT.toFixed(4)} dy={-16} color={tokens.color.success} />
      <Annotation x={1} y={GAP_PERTURBED} text={GAP_PERTURBED.toFixed(4)} dy={-16} color={tokens.color.danger} />
    </Plot>
  );
}
