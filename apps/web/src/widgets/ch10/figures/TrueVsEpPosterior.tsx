import { clutterEpFit, clutterEpInit, linspace, normalPdf, pcg32, standardNormal, trapz } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260917;
const TRUE_THETA = 2.5;
const MODEL = { clutterWeight: 0.4, clutterVariance: 9 };
const PRIOR = { mean: [0], variance: 25 };
const DOMAIN: readonly [number, number] = [-8, 8];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 200);

function syntheticData(): number[] {
  const rng = pcg32(SEED);
  const points: number[] = [];
  for (let i = 0; i < 5; i++) points.push(TRUE_THETA + standardNormal(rng));
  for (let i = 0; i < 3; i++) points.push(6 * standardNormal(rng));
  return points;
}

const DATA = syntheticData();

function trueUnnormalisedPosterior(theta: number): number {
  let density = normalPdf(theta, { mu: PRIOR.mean[0]!, sigma2: PRIOR.variance });
  for (const x of DATA) {
    density *=
      (1 - MODEL.clutterWeight) * normalPdf(x, { mu: theta, sigma2: 1 }) + MODEL.clutterWeight * normalPdf(x, { mu: 0, sigma2: MODEL.clutterVariance });
  }
  return density;
}

export default function TrueVsEpPosterior() {
  const tokens = useResolvedTokens();

  const unnormalised = GRID.map(trueUnnormalisedPosterior);
  const z = trapz(unnormalised, GRID);
  const trueCurve = GRID.map((theta, i) => [theta, unnormalised[i]! / z] as const);

  const initial = clutterEpInit(PRIOR, DATA.length);
  const history = clutterEpFit(
    DATA.map((x) => [x] as const),
    initial,
    MODEL,
    1,
  );
  const epPosterior = history[history.length - 1]!.posterior;
  const epCurve = GRID.map((theta) => [theta, normalPdf(theta, { mu: epPosterior.mean[0]!, sigma2: epPosterior.variance })] as const);

  return (
    <Plot height={220} xDomain={DOMAIN} yDomain={[0, Math.max(...trueCurve.map((p) => p[1])) * 1.15]} label="The true (numerically normalised) posterior against EP's Gaussian approximation">
      <Axes x={{ label: 'theta' }} y={{ label: 'density' }} grid />
      <Curve points={trueCurve} color={tokens.color.ink} width={2} />
      <Curve points={epCurve} color={tokens.series[0]!} width={2} dash="dashed" />
      <Legend
        entries={[
          { label: 'true posterior (grid-normalised)', color: tokens.color.ink, mark: 'line' },
          { label: 'EP Gaussian', color: tokens.series[0]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
