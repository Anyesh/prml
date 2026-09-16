import { linspace, logisticLocalBound, sigmoid } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const XI = 2.5;
const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 120);

export default function WorkedExampleBoundGap() {
  const tokens = useResolvedTokens();
  const sigmoidPoints = GRID.map((x) => [x, sigmoid(x)] as const);
  const boundPoints = GRID.map((x) => [x, logisticLocalBound(x, XI)] as const);
  const marked = [
    { x: -XI, y: sigmoid(-XI) },
    { x: 0, y: sigmoid(0) },
    { x: 0, y: logisticLocalBound(0, XI) },
    { x: XI, y: sigmoid(XI) },
  ];

  return (
    <Plot height={240} xDomain={DOMAIN} yDomain={[0, 1]} label="The worked example's four numbers marked on the curve and its bound at xi = 2.5">
      <Axes x={{ label: 'x' }} y={{ label: 'sigma(x)' }} grid zeroLine />
      <Curve points={sigmoidPoints} color={tokens.color.ink} width={2} />
      <Curve points={boundPoints} color={tokens.color.accent} width={2} dash="dashed" />
      <ScatterField points={marked.map((p) => ({ x: p.x, y: p.y, color: tokens.color.danger, size: 4.5 }))} />
    </Plot>
  );
}
