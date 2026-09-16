import { linspace, normalPdf, trapz } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const CAVITY = { mu: 0, sigma2: 9 };
const MODEL = { clutterWeight: 0.4, clutterVariance: 9 };
const OBSERVATION = 3;
const DOMAIN: readonly [number, number] = [-8, 10];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 200);

function trueFactor(theta: number): number {
  return (1 - MODEL.clutterWeight) * normalPdf(OBSERVATION, { mu: theta, sigma2: 1 }) + MODEL.clutterWeight * normalPdf(OBSERVATION, { mu: 0, sigma2: MODEL.clutterVariance });
}

export default function MomentMatchingIllustration() {
  const tokens = useResolvedTokens();

  const unnormalisedTilted = GRID.map((theta) => normalPdf(theta, CAVITY) * trueFactor(theta));
  const z = trapz(unnormalisedTilted, GRID);
  const tiltedDensity = unnormalisedTilted.map((v) => v / z);

  let mean = 0;
  for (let i = 0; i < GRID.length; i++) mean += GRID[i]! * tiltedDensity[i]!;
  mean *= (GRID[1]! - GRID[0]!);
  let variance = 0;
  for (let i = 0; i < GRID.length; i++) variance += (GRID[i]! - mean) ** 2 * tiltedDensity[i]!;
  variance *= (GRID[1]! - GRID[0]!);

  const cavityCurve = GRID.map((theta) => [theta, normalPdf(theta, CAVITY)] as const);
  const tiltedCurve = GRID.map((theta, i) => [theta, tiltedDensity[i]!] as const);
  const matchedCurve = GRID.map((theta) => [theta, normalPdf(theta, { mu: mean, sigma2: variance })] as const);

  return (
    <Plot height={220} xDomain={DOMAIN} yDomain={[0, Math.max(...tiltedDensity) * 1.15]} label="The cavity, the true (non-Gaussian) tilted density, and the Gaussian that matches its mean and variance">
      <Axes x={{ label: 'theta' }} y={{ label: 'density' }} grid />
      <Curve points={cavityCurve} color={tokens.color.inkFaint} width={1.5} dash="dashed" />
      <Curve points={tiltedCurve} color={tokens.color.ink} width={2} />
      <Curve points={matchedCurve} color={tokens.series[0]!} width={2} dash="dashed" />
      <Legend
        entries={[
          { label: 'cavity q^{\\n}', color: tokens.color.inkFaint, mark: 'dashed-line' },
          { label: 'true tilted density', color: tokens.color.ink, mark: 'line' },
          { label: 'moment-matched Gaussian', color: tokens.series[0]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
