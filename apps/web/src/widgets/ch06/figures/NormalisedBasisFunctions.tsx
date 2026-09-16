import { gaussianBasis, linspace } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const CENTRES = linspace(0, 1, 7);
const SCALE = 0.12;
const GRID = linspace(-0.2, 1.2, 200);
const PHI = gaussianBasis(CENTRES, SCALE, { bias: false });

const RAW_ROWS = GRID.map((x) => PHI(x));
const NORMALISED_ROWS = RAW_ROWS.map((row) => {
  const total = row.reduce((s, v) => s + v, 0);
  return total > 1e-12 ? row.map((v) => v / total) : row;
});

export default function NormalisedBasisFunctions() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-0.2, 1.2]} yDomain={[0, 1.05]} label="Raw Gaussian basis functions">
        <Axes x={{ label: 'x' }} y={{ label: 'phi_j(x)' }} grid />
        {CENTRES.map((_, j) => (
          <Curve key={j} points={GRID.map((x, i) => [x, RAW_ROWS[i]![j]!] as const)} color={tokens.series[j % tokens.series.length]!} width={1.5} />
        ))}
      </Plot>
      <Plot height={200} xDomain={[-0.2, 1.2]} yDomain={[0, 1.05]} label="The same functions normalised to sum to one at every x">
        <Axes x={{ label: 'x' }} y={{ label: 'h_j(x)' }} grid />
        {CENTRES.map((_, j) => (
          <Curve key={j} points={GRID.map((x, i) => [x, NORMALISED_ROWS[i]![j]!] as const)} color={tokens.series[j % tokens.series.length]!} width={1.5} />
        ))}
      </Plot>
      <p className="widget-readout">
        Raw bumps fade to nearly nothing past the outermost centres, at x below 0 or above 1
        here. Dividing each by the row sum, as (6.41) does, keeps every point covered by
        weights that add to exactly one, at the cost of the tails no longer looking Gaussian.
      </p>
    </div>
  );
}
