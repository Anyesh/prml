import { gaussianBasis, linspace, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SCALE = 0.06;
const GRID = linspace(0, 1, 200);

const rng = pcg32(20260701);
const CLUSTERED_DATA = [
  ...Array.from({ length: 9 }, () => 0.15 + 0.05 * rng.next()),
  ...Array.from({ length: 9 }, () => 0.75 + 0.08 * rng.next()),
].sort((a, b) => a - b);

const EVEN_CENTRES = linspace(0.05, 0.95, CLUSTERED_DATA.length);

function bumps(centres: readonly number[]) {
  const phi = gaussianBasis(centres, SCALE, { bias: false });
  return centres.map((_, j) => GRID.map((x) => [x, phi(x)[j]!] as const));
}

export default function CentresOnDataVsGrid() {
  const tokens = useResolvedTokens();
  const onData = bumps(CLUSTERED_DATA);
  const onGrid = bumps(EVEN_CENTRES);

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[0, 1]} yDomain={[0, 1.1]} label="Basis functions centred on the data itself">
        <Axes x={{ label: 'x' }} y={{ label: 'phi(x)' }} grid />
        {onData.map((points, i) => (
          <Curve key={i} points={points} color={tokens.series[0]!} width={1.2} />
        ))}
        <ScatterField points={CLUSTERED_DATA.map((x, i) => ({ x, y: 0, id: i }))} color={tokens.color.ink} size={3} />
      </Plot>
      <Plot height={200} xDomain={[0, 1]} yDomain={[0, 1.1]} label="The same number of basis functions on a fixed evenly spaced grid">
        <Axes x={{ label: 'x' }} y={{ label: 'phi(x)' }} grid />
        {onGrid.map((points, i) => (
          <Curve key={i} points={points} color={tokens.series[1]!} width={1.2} />
        ))}
        <ScatterField points={CLUSTERED_DATA.map((x, i) => ({ x, y: 0, id: i }))} color={tokens.color.ink} size={3} />
      </Plot>
      <p className="widget-readout">
        Eighteen bumps either way. Placed on the data, every bump sits where a point does;
        spread evenly, most of the middle third gets covered while the data never appears
        there, which is exactly the mismatch 3.6 raised as an argument for a fixed basis.
      </p>
    </div>
  );
}
