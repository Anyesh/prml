import { useMemo, useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../../widgets.css';

const MU1 = 0;
const MU2 = 3;
const SIGMA2 = 1;
const P1 = 0.5;
const GRID = linspace(-5, 8, 400);

function joint1(x: number): number {
  return P1 * normalPdf(x, { mu: MU1, sigma2: SIGMA2 });
}
function joint2(x: number): number {
  return (1 - P1) * normalPdf(x, { mu: MU2, sigma2: SIGMA2 });
}

function riskMinimisingBoundary(l12: number, l21: number): number {
  let best = GRID[0]!;
  let bestGap = Infinity;
  for (const x of GRID) {
    const gap = Math.abs(l21 * joint2(x) - l12 * joint1(x));
    if (gap < bestGap) {
      bestGap = gap;
      best = x;
    }
  }
  return best;
}

export default function LossMatrixShift() {
  const [logRatio, setLogRatio] = useState(0);
  const tokens = useResolvedTokens();

  const l12 = Math.exp(logRatio);
  const boundary = useMemo(() => riskMinimisingBoundary(l12, 1), [l12]);
  const equalBoundary = riskMinimisingBoundary(1, 1);

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-5, 8]} yDomain={[0, 0.25]} label="Risk-minimising boundary as the loss ratio changes">
        <Axes x={{ label: 'x' }} y={{ label: 'p(x, C)' }} grid />
        <Curve points={GRID.map((x) => [x, joint1(x)] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={GRID.map((x) => [x, joint2(x)] as const)} color={tokens.series[1]!} width={2} />
        <Rule x={equalBoundary} color={tokens.color.inkFaint} label="equal loss" />
        <Rule x={boundary} color={tokens.color.danger} label="risk-minimising" />
      </Plot>
      <Slider label="log(L12 / L21)" value={logRatio} onChange={setLogRatio} min={-4} max={4} step={0.1} />
      <p className="widget-readout">
        {`L12/L21 = ${l12.toFixed(2)} moves the boundary from x=${equalBoundary.toFixed(2)} to x=${boundary.toFixed(2)}: raising the cost of missing C1 pushes the boundary toward C2, accepting more false C2 calls to avoid ever missing a true C1.`}
      </p>
    </div>
  );
}
