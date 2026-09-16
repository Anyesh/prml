import { interpolatedEnergy, normalPdf } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIGMA_G = 3;
const SIGMA_E = 1;
const ALPHAS = [0, 0.2, 0.4, 0.6, 0.8, 1];

function energyOf(sigma2: number) {
  return (z: number) => (z * z) / (2 * sigma2);
}

export default function ChainingIntermediateDistributions() {
  const tokens = useResolvedTokens();
  const eG = energyOf(SIGMA_G * SIGMA_G);
  const eE = energyOf(SIGMA_E * SIGMA_E);

  const curves = ALPHAS.map((alpha) => {
    const energyAlpha = interpolatedEnergy(alpha, eG, eE);
    // energyAlpha(z) = z^2 / (2 * sigma2) for every alpha here, so sigma2 is recoverable
    // from a single evaluation at z = 1 without re-deriving the interpolation by hand.
    const sigma2 = 1 / (2 * energyAlpha(1));
    const points: [number, number][] = Array.from({ length: 150 }, (_, i) => {
      const z = -6 + (12 * i) / 149;
      return [z, normalPdf(z, { mu: 0, sigma2 })];
    });
    return { alpha, points };
  });

  return (
    <div className="widget-grid">
      <Plot width={420} height={260} xDomain={[-6, 6]} yDomain={[0, 0.42]} label="A chain of intermediate distributions interpolating from a wide Gaussian to a narrow one">
        <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
        {curves.map((c, i) => (
          <Curve key={c.alpha} points={c.points} color={tokens.series[i % tokens.series.length]!} width={c.alpha === 0 || c.alpha === 1 ? 2.5 : 1.5} />
        ))}
      </Plot>
      <p className="widget-readout">
        Six steps of PRML 11.75's linear energy interpolation, alpha = 0 (proposal) to 1 (target): every adjacent
        pair is close enough that an importance-sampling ratio between them is cheap to trust.
      </p>
    </div>
  );
}
