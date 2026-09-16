import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(-3, 8, 400);
const P1 = 0.5;

function classDensity1(x: number): number {
  return 0.5 * normalPdf(x, { mu: -1, sigma2: 0.5 }) + 0.5 * normalPdf(x, { mu: 1, sigma2: 0.5 });
}
function classDensity2(x: number): number {
  return normalPdf(x, { mu: 4, sigma2: 1.5 });
}

function posterior1(x: number): number {
  const j1 = P1 * classDensity1(x);
  const j2 = (1 - P1) * classDensity2(x);
  return j1 / (j1 + j2);
}

function decisionBoundary(): number {
  let best = GRID[0]!;
  let bestGap = Infinity;
  for (const x of GRID) {
    const gap = Math.abs(posterior1(x) - 0.5);
    if (gap < bestGap) {
      bestGap = gap;
      best = x;
    }
  }
  return best;
}

const BOUNDARY = decisionBoundary();
const LEFT_MODE_INFLUENCE = posterior1(-1) - posterior1(-3);

export default function ClassDensityVsPosterior() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-3, 8]} yDomain={[0, 0.5]} label="Two class-conditional densities, one of them two-humped">
        <Axes x={{ label: 'x' }} y={{ label: 'p(x | C)' }} grid />
        <Curve points={GRID.map((x) => [x, classDensity1(x)] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={GRID.map((x) => [x, classDensity2(x)] as const)} color={tokens.series[1]!} width={2} />
        <Legend entries={[{ label: 'p(x|C1), two modes', color: tokens.series[0]!, mark: 'line' }, { label: 'p(x|C2)', color: tokens.series[1]!, mark: 'line' }]} placement="top-left" />
      </Plot>
      <Plot height={200} xDomain={[-3, 8]} yDomain={[0, 1]} label="Posterior probability of class 1">
        <Axes x={{ label: 'x' }} y={{ label: 'p(C1 | x)' }} grid />
        <Curve points={GRID.map((x) => [x, posterior1(x)] as const)} color={tokens.color.accent} width={2} />
        <Rule x={BOUNDARY} color={tokens.color.ink} label="decision boundary" />
      </Plot>
      <p className="widget-readout">
        {`p(C1|x) is a single smooth step, even though p(x|C1) has a second mode near x=-1 that barely changes it (posterior moves by only ${LEFT_MODE_INFLUENCE.toFixed(3)} across that whole hump). A model spending capacity on that mode has nothing to show for it at decision time.`}
      </p>
    </div>
  );
}
