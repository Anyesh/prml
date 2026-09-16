import { mvnConditional, normalPdf } from '@prml/math';
import { Annotation, Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const RHO = 0.85;
const Z2 = 1.4;

export default function ConditionalSlice() {
  const tokens = useResolvedTokens();
  const cov = [
    [1, RHO],
    [RHO, 1],
  ];
  const conditional = mvnConditional({ mean: [0, 0], cov }, new Map([[1, Z2]]));
  const mu = conditional.mean[0]!;
  const sigma2 = conditional.cov[0]![0]!;

  const points: [number, number][] = Array.from({ length: 200 }, (_, i) => {
    const z1 = -4 + (8 * i) / 199;
    return [z1, normalPdf(z1, { mu, sigma2 })];
  });

  return (
    <div className="widget-grid">
      <Plot width={420} height={220} xDomain={[-4, 4]} yDomain={[0, 1]} label="The exact conditional distribution one Gibbs step draws from">
        <Axes x={{ label: 'z1' }} y={{ label: 'density' }} grid />
        <Curve points={points} color={tokens.series[2]!} width={2.5} />
        <Rule x={mu} color={tokens.color.inkMuted} label="mean" />
        <Annotation x={mu} y={normalPdf(mu, { mu, sigma2 })} text={`mean ${mu.toFixed(2)}, sd ${Math.sqrt(sigma2).toFixed(2)}`} dy={-14} plate />
      </Plot>
      <p className="widget-readout">
        With z2 = {Z2} fixed and correlation {RHO}, z1's conditional is N({mu.toFixed(2)},{' '}
        {sigma2.toFixed(3)}): a step this narrow is why strong correlation makes the ridge slow to cross.
      </p>
    </div>
  );
}
