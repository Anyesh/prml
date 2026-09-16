import { useMemo } from 'react';
import { dirichletPdf, evalGrid, linspace } from '@prml/math';
import { Curve, Heatmap, Plot, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(0, 1, 60);
const TRIANGLE = [
  [0, 0],
  [1, 0],
  [0, 1],
  [0, 0],
] as const;
const CONCENTRATIONS = [0.1, 1, 10];

function Panel({ alpha, tokens }: { alpha: number; tokens: ReturnType<typeof useResolvedTokens> }) {
  const field = useMemo(
    () =>
      evalGrid(GRID, GRID, (mu1, mu2) => {
        const mu3 = 1 - mu1 - mu2;
        return mu3 < 0 ? 0 : dirichletPdf([mu1, mu2, mu3], { alpha: [alpha, alpha, alpha] });
      }),
    [alpha],
  );
  const fill = useMemo(() => {
    let peak = 0;
    for (const row of field.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak], tokens.sequential);
  }, [field, tokens.sequential]);

  return (
    <Plot height={150} xDomain={[0, 1]} yDomain={[0, 1]} equalAspect label={`Dirichlet density, alpha=${alpha} for all three categories`}>
      <Heatmap data={field} interpolator={fill} />
      <Curve points={TRIANGLE} color={tokens.color.ink} width={1} />
    </Plot>
  );
}

export default function DirichletCorners() {
  const tokens = useResolvedTokens();
  return (
    <div className="widget-grid">
      {CONCENTRATIONS.map((a) => (
        <div key={a}>
          <Panel alpha={a} tokens={tokens} />
          <p className="widget-readout">{`α = ${a}`}</p>
        </div>
      ))}
    </div>
  );
}
