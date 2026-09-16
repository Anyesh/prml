import { evalGrid, gammaPdf, normalPdf, pcg32, standardNormal, univariateGaussianVbFit } from '@prml/math';
import { Axes, ContourField, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const RNG = pcg32(20261001);
const TRUE_MU = 1.4;
const TRUE_SIGMA = 0.6;
const DATA = Array.from({ length: 12 }, () => TRUE_MU + TRUE_SIGMA * standardNormal(RNG));

const PRIOR = { mu0: 0, lambda0: 1e-3, a0: 1e-3, b0: 1e-3 };
const HISTORY = univariateGaussianVbFit(DATA, PRIOR, { qMu: { mu: 0, lambda: 1 }, qTau: { a: 1, b: 1 } }, 10);

const MU_GRID = Array.from({ length: 70 }, (_, i) => -1 + (i * 4) / 69);
const TAU_GRID = Array.from({ length: 70 }, (_, i) => 0.1 + (i * 8) / 69);

function jointField(qMu: { mu: number; lambda: number }, qTau: { a: number; b: number }) {
  return evalGrid(MU_GRID, TAU_GRID, (mu, tau) => normalPdf(mu, { mu: qMu.mu, sigma2: 1 / qMu.lambda }) * gammaPdf(tau, { shape: qTau.a, rate: qTau.b }));
}

const INITIAL_FIELD = jointField(HISTORY[0]!.qMu, HISTORY[0]!.qTau);
const FINAL_FIELD = jointField(HISTORY[HISTORY.length - 1]!.qMu, HISTORY[HISTORY.length - 1]!.qTau);

export default function UnivariateGaussianVbJoint() {
  const tokens = useResolvedTokens();
  return (
    <Plot width={340} height={300} xDomain={[-1, 3]} yDomain={[0.1, 8.1]} label="q(mu)q(tau) before the first sweep against the converged factorisation">
      <Axes x={{ label: 'mu' }} y={{ label: 'tau' }} grid />
      <ContourField data={INITIAL_FIELD} levelCount={4} color={tokens.color.inkMuted} />
      <ContourField data={FINAL_FIELD} levelCount={5} color={tokens.color.accent} />
    </Plot>
  );
}
