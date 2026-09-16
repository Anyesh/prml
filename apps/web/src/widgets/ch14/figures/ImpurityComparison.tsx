import { crossEntropyImpurity, giniImpurity, misclassificationImpurity } from '@prml/math';
import { Axes, FunctionCurve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

function at(p: number, fn: (proportions: readonly number[]) => number): number {
  return fn([p, 1 - p]);
}

export default function ImpurityComparison() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[0, 1]} yDomain={[0, 1]} label="Three impurity measures for a two-class region, against the proportion in class 1">
      <Axes x={{ label: 'p (class 1 proportion)' }} y={{ label: 'impurity' }} grid />
      <FunctionCurve f={(p) => at(p, giniImpurity)} color={tokens.series[0]!} width={2} />
      <FunctionCurve f={(p) => at(p, crossEntropyImpurity) / (2 * Math.LN2)} color={tokens.series[1]!} width={2} />
      <FunctionCurve f={(p) => at(p, misclassificationImpurity)} color={tokens.series[2]!} width={2} />
      <Legend
        entries={[
          { label: 'Gini', color: tokens.series[0]!, mark: 'line' },
          { label: 'cross-entropy (rescaled)', color: tokens.series[1]!, mark: 'line' },
          { label: 'misclassification', color: tokens.series[2]!, mark: 'line' },
        ]}
      />
    </Plot>
  );
}
