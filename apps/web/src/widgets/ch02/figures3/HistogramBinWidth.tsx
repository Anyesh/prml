import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { DATA, DOMAIN_HI, DOMAIN_LO, truePdf } from '../nonparametricData';
import { linspace } from '@prml/math';
import '../../widgets.css';

const GRID = linspace(DOMAIN_LO, DOMAIN_HI, 160);
const WIDTHS = [0.08, 0.35, 1.2];

function histogram(delta: number) {
  const binCount = Math.max(1, Math.round((DOMAIN_HI - DOMAIN_LO) / delta));
  const counts = new Array(binCount).fill(0);
  for (const x of DATA) {
    const idx = Math.min(binCount - 1, Math.max(0, Math.floor((x - DOMAIN_LO) / delta)));
    counts[idx]++;
  }
  const points: (readonly [number, number])[] = [];
  for (let i = 0; i < binCount; i++) {
    const left = DOMAIN_LO + i * delta;
    const density = counts[i] / (DATA.length * delta);
    points.push([left, density], [left + delta, density]);
  }
  return points;
}

export default function HistogramBinWidth() {
  const tokens = useResolvedTokens();
  return (
    <div className="widget-grid">
      {WIDTHS.map((delta) => (
        <div key={delta}>
          <Plot height={160} xDomain={[DOMAIN_LO, DOMAIN_HI]} yDomain={[0, 0.6]} label={`Histogram with bin width ${delta}`}>
            <Axes x={{ label: 'x' }} y={false} grid />
            <Curve points={GRID.map((x) => [x, truePdf(x)] as const)} color={tokens.color.inkFaint} dash="dashed" width={1} />
            <Curve points={histogram(delta)} color={tokens.color.accent} width={1.75} />
          </Plot>
          <p className="widget-readout">{`Δ = ${delta}`}</p>
        </div>
      ))}
    </div>
  );
}
