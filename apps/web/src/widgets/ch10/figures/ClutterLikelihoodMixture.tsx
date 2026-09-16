import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const THETA = 2.5;
const W = 0.4;
const A = 9;
const DOMAIN: readonly [number, number] = [-10, 10];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 160);

function signal(x: number): number {
  return (1 - W) * normalPdf(x, { mu: THETA, sigma2: 1 });
}
function clutter(x: number): number {
  return W * normalPdf(x, { mu: 0, sigma2: A });
}

export default function ClutterLikelihoodMixture() {
  const tokens = useResolvedTokens();
  return (
    <Plot height={240} xDomain={DOMAIN} yDomain={[0, 0.42]} label="The clutter likelihood (10.209) as its two weighted components, theta fixed at 2.5">
      <Axes x={{ label: 'x' }} y={{ label: 'density' }} grid />
      <Curve points={GRID.map((x) => [x, signal(x)] as const)} color={tokens.color.accent} width={2} />
      <Curve points={GRID.map((x) => [x, clutter(x)] as const)} color={tokens.color.inkMuted} width={2} dash="dashed" />
      <Curve points={GRID.map((x) => [x, signal(x) + clutter(x)] as const)} color={tokens.color.ink} width={1.5} dash="dotted" />
    </Plot>
  );
}
