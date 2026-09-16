import { useMemo, useState } from 'react';
import { linspace } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const EPS_GRID = linspace(0.001, 1, 200);
const REFERENCE_D = [1, 2, 5, 20];

function volumeFraction(d: number, eps: number): number {
  return 1 - (1 - eps) ** d;
}

export default function VolumeConcentrationExplorer() {
  const [d, setD] = useState(10);
  const tokens = useResolvedTokens();

  const focusCurve = useMemo(() => EPS_GRID.map((eps) => [eps, volumeFraction(d, eps)] as const), [d]);
  const referenceCurves = useMemo(
    () => REFERENCE_D.map((dr) => EPS_GRID.map((eps) => [eps, volumeFraction(dr, eps)] as const)),
    [],
  );
  const at10Percent = volumeFraction(d, 0.1);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, 1]} yDomain={[0, 1]} label="Fraction of a D-dimensional sphere's volume within epsilon of its surface">
        <Axes x={{ label: 'epsilon' }} y={{ label: 'volume fraction' }} grid />
        {referenceCurves.map((pts, i) => (
          <Curve key={REFERENCE_D[i]} points={pts} color={tokens.color.inkFaint} dash="dashed" width={1} />
        ))}
        <Curve points={focusCurve} color={tokens.color.accent} width={2.5} />
        <Rule x={0.1} color={tokens.color.danger} label="eps = 0.1" />
        <Legend entries={[{ label: `D = ${d} (highlighted)`, color: tokens.color.accent, mark: 'line' }, { label: 'D = 1, 2, 5, 20 (reference)', color: tokens.color.inkFaint, mark: 'dashed-line' }]} placement="bottom-right" />
      </Plot>
      <Panel columns={1} dense>
        <Slider label="Dimension D" value={d} onChange={(v) => setD(Math.round(v))} min={1} max={100} step={1} scale="log" />
        <p className="widget-readout">
          {`At D=${d}, ${(at10Percent * 100).toFixed(1)}% of the sphere's volume already sits within the outer 10% of its radius. Raising D pushes almost all the volume into that thin shell.`}
        </p>
      </Panel>
    </div>
  );
}
