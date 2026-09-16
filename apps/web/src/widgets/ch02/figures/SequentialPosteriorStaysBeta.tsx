import { betaPdf, betaPosterior, linspace } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const X = linspace(0.005, 0.995, 200);
const PRIOR = { a: 2, b: 2 };
const POSTERIOR = betaPosterior(PRIOR, 1, 0);

export default function SequentialPosteriorStaysBeta() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[0, 1]} yDomain={[0, 2.2]} label="Prior, likelihood, and posterior after one observation of heads">
        <Axes x={{ label: 'µ' }} y={{ label: 'density (likelihood unnormalised)' }} grid />
        <Curve points={X.map((x) => [x, betaPdf(x, PRIOR)] as const)} color={tokens.color.inkFaint} width={1.75} />
        <Curve points={X.map((x) => [x, x] as const)} color={tokens.color.danger} dash="dashed" width={1.5} />
        <Curve points={X.map((x) => [x, betaPdf(x, POSTERIOR)] as const)} color={tokens.color.accent} width={2} />
        <Legend
          entries={[
            { label: 'prior Beta(2,2)', color: tokens.color.inkFaint, mark: 'line' },
            { label: 'likelihood ∝ µ', color: tokens.color.danger, mark: 'dashed-line' },
            { label: `posterior Beta(${POSTERIOR.a},${POSTERIOR.b})`, color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-left"
        />
      </Plot>
    </div>
  );
}
