import { evalGrid, linspace, mvnConditional, mvnLogPdf, mvnMarginal, normalPdf } from '@prml/math';
import { Axes, ContourField, Curve, Legend, Plot, Rule, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const JOINT = { mean: [0, 0], cov: [[1, 0.6], [0.6, 1]] };
const X_B = 0.7;
const GRID = linspace(-3, 3, 70);
const DENSITY = evalGrid(GRID, GRID, (a, b) => Math.exp(mvnLogPdf([a, b], JOINT)));

const MARGINAL_A = mvnMarginal(JOINT, [0]);
const CONDITIONAL_A = mvnConditional(JOINT, new Map([[1, X_B]]));

export default function PartitionAndConditional() {
  const tokens = useResolvedTokens();
  const fill = sequentialScale([0, Math.exp(mvnLogPdf(JOINT.mean, JOINT))], tokens.sequential);
  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-3, 3]} yDomain={[-3, 3]} equalAspect label="Joint density with a slice at x_b = 0.7">
        <ContourField data={DENSITY} levelCount={6} color={tokens.color.inkFaint} fill={fill} z={-1} />
        <Axes x={{ label: 'x_a' }} y={{ label: 'x_b' }} grid />
        <Rule y={X_B} color={tokens.color.danger} />
      </Plot>
      <Plot height={220} xDomain={[-3, 3]} yDomain={[0, 1]} label="Marginal p(x_a) against conditional p(x_a | x_b = 0.7)">
        <Axes x={{ label: 'x_a' }} y={{ label: 'density' }} grid />
        <Curve points={GRID.map((a) => [a, normalPdf(a, { mu: MARGINAL_A.mean[0]!, sigma2: MARGINAL_A.cov[0]![0]! })] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={GRID.map((a) => [a, normalPdf(a, { mu: CONDITIONAL_A.mean[0]!, sigma2: CONDITIONAL_A.cov[0]![0]! })] as const)} color={tokens.color.danger} width={2} />
        <Legend
          entries={[
            { label: 'marginal p(x_a)', color: tokens.series[0]!, mark: 'line' },
            { label: 'conditional p(x_a|x_b=0.7)', color: tokens.color.danger, mark: 'line' },
          ]}
          placement="top-left"
        />
      </Plot>
    </div>
  );
}
