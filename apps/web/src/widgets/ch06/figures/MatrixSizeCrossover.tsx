import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 6;
const M_VALUES = Array.from({ length: 30 }, (_, i) => i + 1);
const PRIMAL_COST = M_VALUES.map((m) => Math.pow(m, 3));
const DUAL_COST = Math.pow(N, 3);
const CROSSOVER = M_VALUES.find((m) => Math.pow(m, 3) > DUAL_COST)!;

export default function MatrixSizeCrossover() {
  const tokens = useResolvedTokens();
  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[1, 30]} yDomain={[0, Math.pow(30, 3)]} label="Cost of inverting the weight-space matrix against the number of basis functions, with the fixed dual-space cost marked">
        <Axes x={{ label: 'M, number of basis functions' }} y={{ label: 'operations (cubic in matrix size)' }} grid />
        <Rule y={DUAL_COST} color={tokens.color.danger} label={`N³ = ${N}³, fixed by the data`} />
        <Curve points={M_VALUES.map((m, i) => [m, PRIMAL_COST[i]!] as const)} color={tokens.series[0]!} width={2} />
        <Legend
          entries={[
            { label: 'M³, weight space', color: tokens.series[0]!, mark: 'line' },
            { label: 'N³, dual space (fixed)', color: tokens.color.danger, mark: 'dashed-line' },
          ]}
          placement="top-left"
        />
      </Plot>
      <p className="widget-readout">
        {`With N = ${N} training points fixed, the weight-space cost overtakes the dual-space cost once M reaches ${CROSSOVER}. ` +
          `Below that crossing, inverting in weight space is cheaper; above it, the dual formulation wins on cost alone, before it ever offers an infinite feature space.`}
      </p>
    </div>
  );
}
