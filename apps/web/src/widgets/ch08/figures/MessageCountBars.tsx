import { maxSum, sumProduct, type FactorGraph } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRAPH: FactorGraph = {
  variables: [
    { id: 'x1', states: 2 },
    { id: 'x2', states: 3 },
    { id: 'x3', states: 2 },
    { id: 'x4', states: 2 },
  ],
  factors: [
    { id: 'fa', scope: ['x1', 'x3'], table: [1.0, 0.4, 0.9, 1.7] },
    { id: 'fb', scope: ['x2', 'x3'], table: [0.5, 1.2, 1.0, 0.3, 2.1, 0.8] },
    { id: 'fc', scope: ['x3', 'x4'], table: [1.4, 0.6, 0.2, 1.1] },
  ],
};

export default function MessageCountBars() {
  const tokens = useResolvedTokens();
  const sp = sumProduct(GRAPH, 'x3');
  const ms = maxSum(GRAPH, 'x3');
  const bars = [
    { at: 0, value: sp.schedule.length, color: tokens.color.accent },
    { at: 1, value: ms.schedule.length + ms.backtrack.length, color: tokens.color.danger },
  ];

  return (
    <Plot width={260} height={200} xDomain={[-0.6, 1.6]} yDomain={[0, 14]} label="Total steps to finish: sum-product's two full passes against max-sum's one pass plus backtrack">
      <Axes y={{ label: 'steps' }} grid />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
