import { useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, ClickSurface, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-3, 3];
const SIMPLE = { mu: 0, sigma2: 0.09 };
const COMPLEX = { mu: 0, sigma2: 2.25 };
const CURVE_POINTS = linspace(DOMAIN[0], DOMAIN[1], 160);
const PEAK_Y = normalPdf(SIMPLE.mu, SIMPLE) * 1.15;
const INITIAL_X = 0.35;

export default function DatasetBudget() {
  const [x, setX] = useState(INITIAL_X);
  const tokens = useResolvedTokens();

  const clampedX = Math.min(DOMAIN[1], Math.max(DOMAIN[0], x));
  const pSimple = normalPdf(clampedX, SIMPLE);
  const pComplex = normalPdf(clampedX, COMPLEX);
  const simpleWins = pSimple > pComplex;

  return (
    <div>
      <Plot
        height={220}
        xDomain={DOMAIN}
        yDomain={[0, PEAK_Y]}
        label="Two distributions over the space of possible datasets, with a draggable observed dataset"
      >
        <ClickSurface onClick={(nx) => setX(Math.min(DOMAIN[1], Math.max(DOMAIN[0], nx)))} cursor="ew-resize" />
        <Axes x={{ label: 'space of possible datasets', bare: true }} y={false} />
        <Curve
          points={CURVE_POINTS.map((v) => [v, normalPdf(v, SIMPLE)] as const)}
          color={tokens.series[0]!}
          width={2}
        />
        <Curve
          points={CURVE_POINTS.map((v) => [v, normalPdf(v, COMPLEX)] as const)}
          color={tokens.series[1]!}
          width={2}
        />
        <Curve
          points={[[clampedX, 0], [clampedX, PEAK_Y]] as const}
          color={tokens.color.ink}
          dash="dashed"
          width={1.25}
        />
        <ScatterField
          points={[{ x: clampedX, y: 0, color: tokens.color.ink, size: 5 }]}
          onMove={(_, nx) => setX(Math.min(DOMAIN[1], Math.max(DOMAIN[0], nx)))}
          label={() => 'Observed dataset'}
        />
        <Legend
          entries={[
            { label: 'simple model', color: tokens.series[0]!, mark: 'line' },
            { label: 'complex model', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`At this dataset, the simple model assigns probability ${pSimple.toFixed(3)} and the complex model `}
        {`${pComplex.toFixed(3)}. ${simpleWins ? 'The simple model wins' : 'The complex model wins'} here`}
        {simpleWins
          ? ", even though the complex model's curve also covers this point: it spreads its unit area over a much wider range of datasets, so any single one gets a smaller share."
          : ', because this dataset sits too far from what the simple model considers plausible at all.'}
      </p>
    </div>
  );
}
