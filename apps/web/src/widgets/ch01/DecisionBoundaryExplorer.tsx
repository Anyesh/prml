import { useMemo, useState } from 'react';
import { clamp, linspace, normalPdf, trapz } from '@prml/math';
import { Axes, Band, Curve, Legend, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const MU1 = 0;
const MU2 = 3;
const SIGMA2 = 1;
const GRID = linspace(-5, 8, 400);

function joint1(x: number, p1: number): number {
  return p1 * normalPdf(x, { mu: MU1, sigma2: SIGMA2 });
}
function joint2(x: number, p1: number): number {
  return (1 - p1) * normalPdf(x, { mu: MU2, sigma2: SIGMA2 });
}

function mistakeProbability(boundary: number, p1: number): number {
  const belowXs = GRID.filter((x) => x < boundary);
  const aboveXs = GRID.filter((x) => x >= boundary);
  const belowErr = belowXs.map((x) => joint2(x, p1));
  const aboveErr = aboveXs.map((x) => joint1(x, p1));
  return trapz(belowErr, belowXs) + trapz(aboveErr, aboveXs);
}

function bayesBoundary(p1: number): number {
  let best = GRID[0]!;
  let bestGap = Infinity;
  for (const x of GRID) {
    const gap = Math.abs(joint1(x, p1) - joint2(x, p1));
    if (gap < bestGap) {
      bestGap = gap;
      best = x;
    }
  }
  return best;
}

export default function DecisionBoundaryExplorer() {
  const [boundary, setBoundary] = useState(1.5);
  const [p1, setP1] = useState(0.5);
  const tokens = useResolvedTokens();

  const j1Curve = useMemo(() => GRID.map((x) => [x, joint1(x, p1)] as const), [p1]);
  const j2Curve = useMemo(() => GRID.map((x) => [x, joint2(x, p1)] as const), [p1]);
  const mistakeBand = useMemo(() => {
    return GRID.map((x) => {
      const err = x < boundary ? joint2(x, p1) : joint1(x, p1);
      return [x, 0, err] as const;
    });
  }, [boundary, p1]);

  const pMistake = mistakeProbability(boundary, p1);
  const optimal = bayesBoundary(p1);
  const pMistakeOptimal = mistakeProbability(optimal, p1);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[-5, 8]} yDomain={[0, 0.5]} label="Joint densities p(x,C1) and p(x,C2), with the shaded region the probability of a mistake">
        <Axes x={{ label: 'x' }} y={{ label: 'p(x, C)' }} grid />
        <Band points={mistakeBand} color={tokens.color.danger} opacity={0.35} />
        <Curve points={j1Curve} color={tokens.series[0]!} width={2} />
        <Curve points={j2Curve} color={tokens.series[1]!} width={2} />
        <Rule x={boundary} color={tokens.color.ink} label="your boundary" />
        <ScatterField
          points={[{ x: boundary, y: 0.48, id: 'handle' }]}
          color={tokens.color.ink}
          shape="triangle"
          size={6}
          onMove={(_, x) => setBoundary(clamp(x, -5, 8))}
          label={() => 'drag to move the decision boundary'}
        />
        <Legend entries={[{ label: 'p(x, C1)', color: tokens.series[0]!, mark: 'line' }, { label: 'p(x, C2)', color: tokens.series[1]!, mark: 'line' }, { label: 'mistakes', color: tokens.color.danger, mark: 'swatch' }]} placement="top-right" />
      </Plot>
      <Panel columns={1} dense>
        <Slider label="Prior p(C1)" value={p1} onChange={setP1} min={0.05} max={0.95} step={0.01} />
        <p className="widget-readout">
          {`p(mistake) at your boundary: ${pMistake.toFixed(4)}. Best possible, at x=${optimal.toFixed(2)} where the two curves cross: ${pMistakeOptimal.toFixed(4)}. Drag onto the crossing point and the shaded area is as small as it can get.`}
        </p>
      </Panel>
    </div>
  );
}
