import { eigSym, exactHessian, linspace, matScale } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { BETA, REG_DATASET, REG_SPEC, W_MAP } from '../bayesianToyNetwork';
import '../../widgets.css';

const BETA_H = matScale(exactHessian(REG_SPEC, W_MAP, REG_DATASET, 'sumSquared'), BETA);
const EIGENVALUES = eigSym(BETA_H).values;
const ALPHA_GRID = linspace(0.01, 5, 60);
const W = EIGENVALUES.length;

function gamma(alpha: number): number {
  let sum = 0;
  for (const lambda of EIGENVALUES) sum += lambda / (alpha + lambda);
  return sum;
}

export default function EffectiveParametersVsAlpha() {
  const tokens = useResolvedTokens();
  const points = ALPHA_GRID.map((alpha) => [alpha, gamma(alpha)] as const);

  return (
    <Plot height={220} xDomain={[0, 5]} yDomain={[0, W]} label="Effective number of parameters against the prior precision α">
      <Axes x={{ label: 'α' }} y={{ label: 'γ' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
