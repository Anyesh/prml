import { irlsFit, norm, pcg32, standardNormal } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEPARATIONS = [0.3, 1.5, 4];
const N_PER_CLASS = 20;
const MAX_ITERATIONS = 15;

function buildDataset(separation: number) {
  const rng = pcg32(20260431);
  const class0 = Array.from({ length: N_PER_CLASS }, () => [1, -separation / 2 + 0.4 * standardNormal(rng)]);
  const class1 = Array.from({ length: N_PER_CLASS }, () => [1, separation / 2 + 0.4 * standardNormal(rng)]);
  return { design: [...class0, ...class1], targets: [...class0.map(() => 0), ...class1.map(() => 1)] };
}

export default function WeightNormDivergence() {
  const tokens = useResolvedTokens();

  const curves = SEPARATIONS.map((separation) => {
    const { design, targets } = buildDataset(separation);
    const fit = irlsFit(design, targets, { maxIterations: MAX_ITERATIONS });
    const norms = [0, ...fit.history.map((h) => norm(h.weights))];
    return norms.map((n, i) => [i, n] as const);
  });
  const maxNorm = Math.max(...curves.flat().map(([, n]) => n));

  return (
    <Plot height={220} xDomain={[0, MAX_ITERATIONS]} yDomain={[0, maxNorm * 1.05]} label="‖w‖ against IRLS iteration at three class separations">
      <Axes x={{ label: 'iteration' }} y={{ label: '‖w‖' }} grid />
      {curves.map((c, i) => (
        <Curve key={i} points={c} color={tokens.series[i]!} width={2} />
      ))}
      <Legend
        entries={SEPARATIONS.map((s, i) => ({ label: `separation ${s}`, color: tokens.series[i]!, mark: 'line' as const }))}
        placement="top-left"
      />
    </Plot>
  );
}
