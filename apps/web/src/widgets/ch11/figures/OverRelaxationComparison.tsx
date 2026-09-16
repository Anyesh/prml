import { mvnConditional, overRelaxationStep, pcg32, standardNormal, type Rng } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const RHO = 0.95;
const COV = [
  [1, RHO],
  [RHO, 1],
];
const MEAN = [0, 0];
const N_SWEEPS = 60;
const ALPHA = -0.8;

function conditionalFor(z: number[], index: number): { mean: number; variance: number } {
  const observed = new Map<number, number>();
  for (let i = 0; i < z.length; i++) if (i !== index) observed.set(i, z[i]!);
  const c = mvnConditional({ mean: MEAN, cov: COV }, observed);
  return { mean: c.mean[0]!, variance: c.cov[0]![0]! };
}

function plainGibbsDistance(rng: Rng): number[] {
  let z = [4, -4];
  const distances = [Math.hypot(z[0]!, z[1]!)];
  for (let s = 0; s < N_SWEEPS; s++) {
    for (const index of [0, 1]) {
      const { mean, variance } = conditionalFor(z, index);
      z = z.slice();
      z[index] = mean + Math.sqrt(variance) * standardNormal(rng);
    }
    distances.push(Math.hypot(z[0]!, z[1]!));
  }
  return distances;
}

function overRelaxedDistance(rng: Rng): number[] {
  let z = [4, -4];
  const distances = [Math.hypot(z[0]!, z[1]!)];
  for (let s = 0; s < N_SWEEPS; s++) {
    for (const index of [0, 1]) {
      const { mean, variance } = conditionalFor(z, index);
      const nu = standardNormal(rng);
      z = z.slice();
      z[index] = overRelaxationStep(z[index]!, mean, variance, ALPHA, nu);
    }
    distances.push(Math.hypot(z[0]!, z[1]!));
  }
  return distances;
}

export default function OverRelaxationComparison() {
  const tokens = useResolvedTokens();
  const plain = plainGibbsDistance(pcg32(2026, 90));
  const relaxed = overRelaxedDistance(pcg32(2026, 91));

  return (
    <div className="widget-grid">
      <Plot width={420} height={240} xDomain={[0, N_SWEEPS]} yDomain={[0, 6]} label="Distance from the origin over sweeps, plain Gibbs against over-relaxation">
        <Axes x={{ label: 'sweep' }} y={{ label: 'distance from origin' }} grid />
        <Curve points={plain.map((d, i) => [i, d] as const)} color={tokens.color.inkMuted} width={2} dash="dashed" />
        <Curve points={relaxed.map((d, i) => [i, d] as const)} color={tokens.series[3]!} width={2.5} />
      </Plot>
      <p className="widget-readout">
        Same start, same correlation rho = {RHO}: over-relaxation (alpha = {ALPHA}) pushes each step to the far side
        of its conditional mean, reaching the equilibrium region in noticeably fewer sweeps than plain Gibbs.
      </p>
    </div>
  );
}
