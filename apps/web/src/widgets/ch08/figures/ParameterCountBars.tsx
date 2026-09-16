import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const K_VALUES = [2, 3, 4, 5, 6, 7];

function fullyConnectedParams(k: number): number {
  return Array.from({ length: k }, (_, i) => 2 ** i).reduce((a, b) => a + b, 0);
}

export default function ParameterCountBars() {
  const tokens = useResolvedTokens();
  const independentBars = K_VALUES.map((k) => ({ at: k, value: k, color: tokens.color.accent }));
  const fullBars = K_VALUES.map((k) => ({ at: k + 0.35, value: fullyConnectedParams(k), color: tokens.color.danger }));

  return (
    <Plot
      width={320}
      height={240}
      xDomain={[1.5, 8]}
      yDomain={[0, fullyConnectedParams(7) * 1.05]}
      label="Free parameters for K binary variables: independent versus fully connected"
    >
      <Axes x={{ label: 'K variables' }} y={{ label: 'free parameters' }} grid />
      <Bars bars={independentBars} thickness={0.3} />
      <Bars bars={fullBars} thickness={0.3} />
    </Plot>
  );
}
