import { linspace, normalPdf, trapz } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const Z = linspace(-6, 6, 400);
const COMPONENT_MU = 2;
const COMPONENT_VAR = 0.5;

function mixturePdf(z: number): number {
  return 0.5 * normalPdf(z, { mu: -COMPONENT_MU, sigma2: COMPONENT_VAR }) + 0.5 * normalPdf(z, { mu: COMPONENT_MU, sigma2: COMPONENT_VAR });
}

function klQtoP(mu: number, sigma2: number): number {
  const integrand = Z.map((z) => {
    const q = normalPdf(z, { mu, sigma2 });
    if (q < 1e-300) return 0;
    return q * (Math.log(q) - Math.log(mixturePdf(z)));
  });
  return trapz(integrand, Z);
}

function localMinimum(muLo: number, muHi: number): { mu: number; sigma2: number } {
  let best = { mu: muLo, sigma2: COMPONENT_VAR, kl: Infinity };
  for (let mu = muLo; mu <= muHi; mu += 0.05) {
    for (let sigma2 = 0.1; sigma2 <= 2; sigma2 += 0.05) {
      const kl = klQtoP(mu, sigma2);
      if (kl < best.kl) best = { mu, sigma2, kl };
    }
  }
  return best;
}

const Q_FORWARD_LEFT = localMinimum(-4, 0);
const Q_FORWARD_RIGHT = localMinimum(0, 4);
const Q_REVERSE_VARIANCE = 0.5 * (COMPONENT_VAR + COMPONENT_MU * COMPONENT_MU) + 0.5 * (COMPONENT_VAR + COMPONENT_MU * COMPONENT_MU);

const P_CURVE: (readonly [number, number])[] = Z.map((z) => [z, mixturePdf(z)]);
const Q_LEFT_CURVE: (readonly [number, number])[] = Z.map((z) => [z, normalPdf(z, Q_FORWARD_LEFT)]);
const Q_RIGHT_CURVE: (readonly [number, number])[] = Z.map((z) => [z, normalPdf(z, Q_FORWARD_RIGHT)]);
const Q_REVERSE_CURVE: (readonly [number, number])[] = Z.map((z) => [z, normalPdf(z, { mu: 0, sigma2: Q_REVERSE_VARIANCE })]);

export default function BimodalKlComparison() {
  const tokens = useResolvedTokens();
  return (
    <Plot height={280} xDomain={[-6, 6]} yDomain={[0, 0.62]} label="A bimodal p against the two KL(q||p) local minima and the single KL(p||q) minimiser">
      <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
      <Curve points={P_CURVE} color={tokens.color.success} width={2} />
      <Curve points={Q_LEFT_CURVE} color={tokens.color.danger} width={1.5} dash="dashed" />
      <Curve points={Q_RIGHT_CURVE} color={tokens.color.danger} width={1.5} dash="dashed" />
      <Curve points={Q_REVERSE_CURVE} color={tokens.color.accent} width={2} dash="dotted" />
    </Plot>
  );
}
