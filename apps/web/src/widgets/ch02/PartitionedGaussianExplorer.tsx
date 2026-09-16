import { useMemo, useState } from 'react';
import { clamp, evalGrid, linspace, mvnConditional, mvnLogPdf, mvnMarginal, normalPdf } from '@prml/math';
import { Axes, ContourField, Curve, Legend, Plot, Rule, sequentialScale, useDrag, useFrame, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'Conditioning and marginalising a 2D Gaussian';
export const caption = 'Reshape the ellipse with the sliders, then drag the handle to change what x2 is conditioned on.';
export const figure = '2.9';

const DOMAIN: readonly [number, number] = [-3.5, 3.5];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 70);
const CURVE_GRID = linspace(DOMAIN[0], DOMAIN[1], 200);
const HANDLE_X = DOMAIN[0] + 0.4;

/** Projects one data point to pixels via `useFrame`, so the drag handle tracks the plot's own scale and margins. */
function ConditioningHandle({ y, onChange, color, bg }: { y: number; onChange: (y: number) => void; color: string; bg: string }) {
  const frame = useFrame();
  const bindings = useDrag({ onDrag: (_x, dragY) => onChange(clamp(dragY, DOMAIN[0], DOMAIN[1])) });
  const [px, py] = frame.toPx(HANDLE_X, y);
  return <circle cx={px} cy={py} r={7} fill={color} stroke={bg} strokeWidth={2} onPointerDown={bindings.onPointerDown} style={bindings.style} />;
}

export default function PartitionedGaussianExplorer() {
  const [sigma1, setSigma1] = useState(1.2);
  const [sigma2, setSigma2] = useState(1);
  const [rho, setRho] = useState(0.6);
  const [xb, setXb] = useState(0.5);
  const tokens = useResolvedTokens();

  const cov = useMemo(() => {
    const c = rho * sigma1 * sigma2;
    return [
      [sigma1 * sigma1, c],
      [c, sigma2 * sigma2],
    ];
  }, [sigma1, sigma2, rho]);
  const params = { mean: [0, 0], cov };

  const density = useMemo(
    () => evalGrid(GRID, GRID, (a, b) => Math.exp(mvnLogPdf([a, b], { mean: [0, 0], cov }))),
    [cov],
  );
  const fill = useMemo(() => {
    let peak = 0;
    for (const row of density.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak], tokens.sequential);
  }, [density, tokens.sequential]);

  const marginalA = mvnMarginal(params, [0]);
  const marginalB = mvnMarginal(params, [1]);
  const conditionalA = mvnConditional(params, new Map([[1, xb]]));

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Joint density with a draggable conditioning line at x2 = xb">
        <ContourField data={density} levelCount={7} color={tokens.color.inkFaint} fill={fill} z={-1} />
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        <Rule y={xb} color={tokens.color.danger} width={1.5} />
        <ConditioningHandle y={xb} onChange={setXb} color={tokens.color.danger} bg={tokens.color.bg} />
      </Plot>

      <Plot height={280} xDomain={DOMAIN} yDomain={[0, 1.4]} label="Marginals and the conditional at the current xb">
        <Axes x={{ label: 'value' }} y={{ label: 'density' }} grid />
        <Curve
          points={CURVE_GRID.map((v) => [v, normalPdf(v, { mu: marginalA.mean[0]!, sigma2: marginalA.cov[0]![0]! })] as const)}
          color={tokens.series[0]!}
          dash="dashed"
          width={1.5}
        />
        <Curve
          points={CURVE_GRID.map((v) => [v, normalPdf(v, { mu: marginalB.mean[0]!, sigma2: marginalB.cov[0]![0]! })] as const)}
          color={tokens.color.inkFaint}
          dash="dashed"
          width={1.5}
        />
        <Curve
          points={CURVE_GRID.map((v) => [v, normalPdf(v, { mu: conditionalA.mean[0]!, sigma2: conditionalA.cov[0]![0]! })] as const)}
          color={tokens.color.danger}
          width={2}
        />
        <Legend
          entries={[
            { label: 'marginal p(x1)', color: tokens.series[0]!, mark: 'dashed-line' },
            { label: 'marginal p(x2)', color: tokens.color.inkFaint, mark: 'dashed-line' },
            { label: 'conditional p(x1|x2=xb)', color: tokens.color.danger, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="σ1" value={sigma1} onChange={setSigma1} min={0.3} max={3} />
        <Slider label="σ2" value={sigma2} onChange={setSigma2} min={0.3} max={3} />
        <Slider label="correlation ρ" value={rho} onChange={setRho} min={-0.9} max={0.9} />
        <p className="widget-readout">
          {`xb = ${xb.toFixed(2)}. Conditional mean ${conditionalA.mean[0]!.toFixed(2)}, variance ${conditionalA.cov[0]![0]!.toFixed(2)}, `}
          {`always tighter than the marginal's ${marginalA.cov[0]![0]!.toFixed(2)} whenever ρ is away from 0.`}
        </p>
      </Panel>
    </div>
  );
}
