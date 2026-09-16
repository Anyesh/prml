import { useMemo, useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, Band, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../../widgets.css';

const MU1 = 0;
const MU2 = 3;
const SIGMA2 = 1;
const GRID = linspace(-5, 8, 400);

function posteriors(x: number): { p1: number; p2: number } {
  const j1 = 0.5 * normalPdf(x, { mu: MU1, sigma2: SIGMA2 });
  const j2 = 0.5 * normalPdf(x, { mu: MU2, sigma2: SIGMA2 });
  const total = j1 + j2;
  return { p1: j1 / total, p2: j2 / total };
}

export default function RejectRegion() {
  const [theta, setTheta] = useState(0.9);
  const tokens = useResolvedTokens();

  const curves = useMemo(() => GRID.map((x) => ({ x, ...posteriors(x) })), []);
  const rejectBand = curves.map((c) => [c.x, 0, Math.max(c.p1, c.p2) < theta ? 1 : 0] as const);
  const rejectFraction = curves.filter((c) => Math.max(c.p1, c.p2) < theta).length / curves.length;

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-5, 8]} yDomain={[0, 1]} label="Posterior probabilities with a reject region shaded where the larger one falls below theta">
        <Axes x={{ label: 'x' }} y={{ label: 'p(C|x)' }} grid />
        <Band points={rejectBand} color={tokens.color.inkFaint} opacity={0.3} />
        <Curve points={curves.map((c) => [c.x, c.p1] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={curves.map((c) => [c.x, c.p2] as const)} color={tokens.series[1]!} width={2} />
      </Plot>
      <Slider label="theta" value={theta} onChange={setTheta} min={0.5} max={1} step={0.01} />
      <p className="widget-readout">
        {`theta = ${theta.toFixed(2)} rejects about ${(rejectFraction * 100).toFixed(0)}% of this grid's inputs. At theta=1 everything is rejected; at theta below 0.5 nothing is, since one posterior always exceeds it in a two-class problem.`}
      </p>
    </div>
  );
}
