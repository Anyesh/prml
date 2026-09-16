import { absoluteResidualError, squaredResidualError } from '@prml/math';
import { Axes, FunctionCurve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

export default function RegressionBoostingComparison() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[-3, 3]} yDomain={[0, 4]} label="Squared against absolute residual error for regression boosting">
      <Axes x={{ label: 'residual' }} y={{ label: 'E' }} grid zeroLine />
      <FunctionCurve f={squaredResidualError} color={tokens.series[0]!} width={2} />
      <FunctionCurve f={absoluteResidualError} color={tokens.series[1]!} width={2} />
      <Legend
        entries={[
          { label: 'squared', color: tokens.series[0]!, mark: 'line' },
          { label: 'absolute', color: tokens.series[1]!, mark: 'line' },
        ]}
      />
    </Plot>
  );
}
