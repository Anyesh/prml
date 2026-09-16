import { kappa, linspace } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(0, 30, 200);

export default function KappaShrinkage() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[0, 30]} yDomain={[0, 1.02]} label="κ(σ²), the factor that rescales the activation before the sigmoid">
      <Axes x={{ label: 'σ²ₐ' }} y={{ label: 'κ(σ²ₐ)' }} grid />
      <Curve points={GRID.map((v) => [v, kappa(v)] as const)} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
