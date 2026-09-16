import { linspace, probit, sigmoid } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LAMBDA = Math.sqrt(Math.PI / 8);
const GRID = linspace(-6, 6, 240);

export default function ProbitVsLogisticSigmoid() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={220} xDomain={[-6, 6]} yDomain={[-0.02, 1.02]} label="The logistic sigmoid against a slope-matched probit">
      <Axes x={{ label: 'a' }} y={{ label: 'probability' }} grid />
      <Curve points={GRID.map((a) => [a, sigmoid(a)] as const)} color={tokens.color.ink} width={2} />
      <Curve points={GRID.map((a) => [a, probit(LAMBDA * a)] as const)} color={tokens.color.accent} width={2} dash="dashed" />
      <Legend
        entries={[
          { label: 'σ(a)', color: tokens.color.ink, mark: 'line' },
          { label: 'Φ(λa), λ² = π/8', color: tokens.color.accent, mark: 'dashed-line' },
        ]}
        placement="top-left"
      />
    </Plot>
  );
}
