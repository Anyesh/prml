import {
  clutterEpFit,
  clutterEpInit,
  clutterMomentMatch,
  isotropicCavity,
  pcg32,
  standardNormal,
  type ClutterModel,
  type IsotropicGaussian,
} from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import type { BarDatum } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const TRUE_THETA = 2.5;
const MODEL: ClutterModel = { clutterWeight: 0.4, clutterVariance: 9 };
const PRIOR: IsotropicGaussian = { mean: [0], variance: 25 };

function syntheticData(): number[] {
  const rng = pcg32(SEED);
  const points: number[] = [];
  for (let i = 0; i < 5; i++) points.push(TRUE_THETA + standardNormal(rng));
  for (let i = 0; i < 3; i++) points.push(6 * standardNormal(rng));
  return points;
}

const DATA = syntheticData();
const INITIAL = clutterEpInit(PRIOR, DATA.length);
const FINAL = clutterEpFit(
  DATA.map((x) => [x]),
  INITIAL,
  MODEL,
  1,
).at(-1)!;

const PROBABILITIES = DATA.map((x, n) => {
  const cavity = isotropicCavity(FINAL.posterior, FINAL.sites[n]!);
  return clutterMomentMatch([x], cavity, MODEL).signalProbability;
});

export default function SignalProbabilityBars() {
  const tokens = useResolvedTokens();
  return (
    <Plot height={240} xDomain={[-0.5, DATA.length - 0.5]} yDomain={[0, 1]} label="Posterior probability each of the eight worked-example points is signal rather than clutter">
      <Axes x={{ label: 'point' }} y={{ label: 'rho_n' }} grid />
      <Bars
        bars={PROBABILITIES.map((rho, n): BarDatum => ({ at: n, value: rho, color: rho > 0.5 ? tokens.color.accent : tokens.color.danger }))}
        thickness={0.6}
      />
    </Plot>
  );
}
