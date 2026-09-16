import { evalGrid, linspace } from '@prml/math';
import { Axes, Heatmap, Plot, ScatterField, useResolvedTokens, withAlpha } from '@prml/viz';
import '../../widgets.css';

const MEANS = [
  [0, 3],
  [-3, -2],
  [3, -2],
] as const;
const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(-6, 6, 140);

function ovrScore(mean: readonly [number, number], x: number, y: number): number {
  return 2 * (mean[0] * x + mean[1] * y) - (mean[0] * mean[0] + mean[1] * mean[1]);
}

function oneVsRestAmbiguous(x: number, y: number): number {
  const claims = MEANS.filter((m) => ovrScore(m, x, y) > 0).length;
  return claims === 1 ? 0 : 1;
}

function oneVsOneWinner(x: number, y: number): number {
  const votes = [0, 0, 0];
  for (let j = 0; j < 3; j++) {
    for (let k = j + 1; k < 3; k++) {
      const mj = MEANS[j]!;
      const mk = MEANS[k]!;
      const midX = (mj[0] + mk[0]) / 2;
      const midY = (mj[1] + mk[1]) / 2;
      const side = (mj[0] - mk[0]) * (x - midX) + (mj[1] - mk[1]) * (y - midY);
      votes[side >= 0 ? j : k]! += 1;
    }
  }
  const maxVotes = Math.max(...votes);
  return votes.filter((v) => v === maxVotes).length > 1 ? 1 : 0;
}

export default function OneVsRestAmbiguity() {
  const tokens = useResolvedTokens();
  const ovr = evalGrid(GRID, GRID, oneVsRestAmbiguous);
  const ovo = evalGrid(GRID, GRID, oneVsOneWinner);
  const ambiguityShade = (t: number) => withAlpha(tokens.color.danger, t > 0.5 ? 0.35 : 0);

  return (
    <div className="widget-grid">
      <Plot height={240} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="One-versus-rest: red shading is claimed by none or by more than one classifier">
        <Heatmap data={ovr} interpolator={ambiguityShade} />
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
        <ScatterField points={MEANS.map((m, i) => ({ x: m[0], y: m[1], id: i, size: 6 }))} color={tokens.color.ink} />
      </Plot>
      <Plot height={240} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="One-versus-one: red shading has no majority winner among the three pairwise votes">
        <Heatmap data={ovo} interpolator={ambiguityShade} />
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
        <ScatterField points={MEANS.map((m, i) => ({ x: m[0], y: m[1], id: i, size: 6 }))} color={tokens.color.ink} />
      </Plot>
    </div>
  );
}
