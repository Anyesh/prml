import { useMemo, useState } from 'react';
import { dirichletMean, dirichletPdf, evalGrid, linspace } from '@prml/math';
import { Axes, Curve, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'Dirichlet on the simplex';
export const caption =
  'Move the three concentrations. Equal and above 1 peaks the density at the centre; below 1 pulls it out to the corners.';
export const figure = '2.5';

const GRID = linspace(0, 1, 90);
const TRIANGLE = [
  [0, 0],
  [1, 0],
  [0, 1],
  [0, 0],
] as const;

export default function DirichletSimplex() {
  const [a1, setA1] = useState(4);
  const [a2, setA2] = useState(4);
  const [a3, setA3] = useState(4);
  const tokens = useResolvedTokens();

  const params = { alpha: [a1, a2, a3] };

  const field = useMemo(
    () =>
      evalGrid(GRID, GRID, (mu1, mu2) => {
        const mu3 = 1 - mu1 - mu2;
        if (mu3 < 0) return 0;
        return dirichletPdf([mu1, mu2, mu3], { alpha: [a1, a2, a3] });
      }),
    [a1, a2, a3],
  );

  const fill = useMemo(() => {
    let peak = 0;
    for (const row of field.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak], tokens.sequential);
  }, [field, tokens.sequential]);

  const mean = dirichletMean(params);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, 1]} yDomain={[0, 1]} equalAspect label="Dirichlet density over the 2-simplex, µ1 by µ2">
        <Heatmap data={field} interpolator={fill} />
        <Curve points={TRIANGLE} color={tokens.color.ink} width={1.5} />
        <Axes x={{ label: 'µ₁' }} y={{ label: 'µ₂' }} />
        <ScatterField points={[{ x: mean[0]!, y: mean[1]!, color: tokens.color.danger, size: 4 }]} label={() => 'mean'} />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="α₁" value={a1} onChange={setA1} min={0.1} max={20} scale="log" />
        <Slider label="α₂" value={a2} onChange={setA2} min={0.1} max={20} scale="log" />
        <Slider label="α₃" value={a3} onChange={setA3} min={0.1} max={20} scale="log" />
        <p className="widget-readout">
          {`Mean at (µ₁,µ₂,µ₃) = (${mean[0]!.toFixed(2)}, ${mean[1]!.toFixed(2)}, ${mean[2]!.toFixed(2)}), always α/α₀.`}
        </p>
      </Panel>
    </div>
  );
}
