import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';

import '../../widgets.css';

const GRID = linspace(-6, 6, 200);
const ICA_DENSITY = GRID.map((z) => [z, 1 / (Math.PI * Math.cosh(z))] as const);
const GAUSSIAN_DENSITY = GRID.map((z) => [z, normalPdf(z, { mu: 0, sigma2: 1 })] as const);

export default function HeavyTailedPrior() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[-6, 6]} yDomain={[0, 0.45]} label="A heavy-tailed ICA source density against a standard Gaussian of the same peak scale">
      <Axes x={{ label: 'z' }} y={{ label: 'p(z)' }} grid />
      <Curve points={ICA_DENSITY} color={tokens.series[0]!} width={2} />
      <Curve points={GAUSSIAN_DENSITY} color={tokens.series[1]!} width={2} dash="dashed" />
      <Legend
        entries={[
          { label: '1 / (pi cosh z)', color: tokens.series[0]!, mark: 'line' },
          { label: 'standard Gaussian', color: tokens.series[1]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
