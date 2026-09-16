import { pcg32, standardNormal, vbGmmResponsibilities, type VbGmmPosterior } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const POSTERIOR: VbGmmPosterior = {
  alpha: [3, 3],
  components: [
    { beta: 1, mean: [-2, 0], scale: [[1, 0], [0, 1]], dof: 4 },
    { beta: 1, mean: [2, 0], scale: [[1, 0], [0, 1]], dof: 4 },
  ],
};
const N = 30;

function syntheticStream(): number[][] {
  const rng = pcg32(SEED);
  return Array.from({ length: N }, () => {
    const cluster = standardNormal(rng) > 0 ? 2 : -2;
    return [cluster + 0.6 * standardNormal(rng), 0.6 * standardNormal(rng)];
  });
}

export default function NaturalParameterAccumulation() {
  const tokens = useResolvedTokens();
  const stream = syntheticStream();

  let alpha0 = 1;
  const points: (readonly [number, number])[] = [[0, alpha0]];
  for (let n = 0; n < N; n++) {
    const r = vbGmmResponsibilities([stream[n]!], POSTERIOR)[0]!;
    alpha0 += r[0]!;
    points.push([n + 1, alpha0]);
  }

  return (
    <Plot height={200} xDomain={[0, N]} yDomain={[0, alpha0 * 1.1]} label="A Dirichlet natural parameter accumulating one responsibility at a time, PRML 10.121">
      <Axes x={{ label: 'points seen' }} y={{ label: 'alpha_1' }} grid />
      <Curve points={points} color={tokens.series[0]!} width={2} />
    </Plot>
  );
}
