import { crossEntropyMarginError, exponentialMarginError, hingeMarginError, misclassificationMarginError } from '@prml/math';
import { Axes, Curve, FunctionCurve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LN2 = Math.log(2);

export default function ErrorFunctionComparison() {
  const tokens = useResolvedTokens();
  const zs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];

  return (
    <Plot height={220} xDomain={[-2, 2]} yDomain={[0, 3]} label="Four error functions of the margin z = t times y(x)">
      <Axes x={{ label: 'z' }} y={{ label: 'E(z)' }} grid zeroLine />
      <FunctionCurve f={exponentialMarginError} color={tokens.series[0]!} width={2} />
      <FunctionCurve f={(z) => crossEntropyMarginError(z) / LN2} color={tokens.series[1]!} width={2} />
      <FunctionCurve f={hingeMarginError} color={tokens.series[2]!} width={2} />
      <Curve points={zs.map((z) => [z, misclassificationMarginError(z)] as const)} color={tokens.color.ink} width={1.5} dash="dotted" />
      <Legend
        entries={[
          { label: 'exponential', color: tokens.series[0]!, mark: 'line' },
          { label: 'cross-entropy / ln 2', color: tokens.series[1]!, mark: 'line' },
          { label: 'hinge', color: tokens.series[2]!, mark: 'line' },
          { label: 'misclassification', color: tokens.color.ink, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
