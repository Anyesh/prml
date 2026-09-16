import { linspace, rvmOptimalSingleAlpha, rvmSingleAlphaLogEvidenceTerm } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LOG_ALPHA = linspace(-5, 6, 200);
const CASE_FINITE = { s: 1, q: 2 };
const CASE_INFINITE = { s: 2, q: 1 };

export default function SparsityMechanism() {
  const tokens = useResolvedTokens();

  const finiteMax = rvmOptimalSingleAlpha(CASE_FINITE.s, CASE_FINITE.q);
  const curveFinite = LOG_ALPHA.map((la) => [la, rvmSingleAlphaLogEvidenceTerm(Math.exp(la), CASE_FINITE.s, CASE_FINITE.q)] as const);
  const curveInfinite = LOG_ALPHA.map((la) => [la, rvmSingleAlphaLogEvidenceTerm(Math.exp(la), CASE_INFINITE.s, CASE_INFINITE.q)] as const);

  return (
    <Plot height={260} xDomain={[-5, 6]} yDomain={[-5, 2]} label="lambda(alpha) for a basis function that survives against one that gets pruned">
      <Axes x={{ label: 'ln α' }} y={{ label: 'λ(α)' }} grid />
      <Curve points={curveFinite} color={tokens.color.accent} width={2} />
      <Curve points={curveInfinite} color={tokens.series[1] ?? tokens.color.ink} width={2} />
      <Legend
        entries={[
          { label: `q² > s: maximum at ln α ≈ ${Math.log(finiteMax).toFixed(2)}`, color: tokens.color.accent, mark: 'line' },
          { label: 'q² < s: λ keeps rising toward α = ∞', color: tokens.series[1] ?? tokens.color.ink, mark: 'line' },
        ]}
        placement="bottom-right"
      />
    </Plot>
  );
}
