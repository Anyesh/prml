import { hingeLoss, linspace, logisticMarginLoss, misclassificationLoss, squaredMarginLoss } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(-2, 2, 200);

export default function HingeLossComparison() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={260} xDomain={[-2, 2]} yDomain={[-0.2, 3]} label="Four error functions plotted against the margin z = t·y(x)">
      <Axes x={{ label: 'z' }} y={{ label: 'E(z)' }} grid zeroLine />
      <Curve points={GRID.map((z) => [z, misclassificationLoss(z)] as const)} color={tokens.color.inkMuted} width={1.5} dash="dotted" />
      <Curve points={GRID.map((z) => [z, squaredMarginLoss(z)] as const)} color={tokens.series[2] ?? tokens.color.ink} width={2} />
      <Curve points={GRID.map((z) => [z, logisticMarginLoss(z)] as const)} color={tokens.series[1] ?? tokens.color.ink} width={2} />
      <Curve points={GRID.map((z) => [z, hingeLoss(z)] as const)} color={tokens.color.accent} width={2.5} />
      <Legend
        entries={[
          { label: 'hinge', color: tokens.color.accent, mark: 'line' },
          { label: 'logistic / ln 2', color: tokens.series[1] ?? tokens.color.ink, mark: 'line' },
          { label: 'squared', color: tokens.series[2] ?? tokens.color.ink, mark: 'line' },
          { label: 'misclassification', color: tokens.color.inkMuted, mark: 'dashed-line' },
        ]}
        placement="top-right"
      />
    </Plot>
  );
}
