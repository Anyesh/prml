import { normalPdf } from '@prml/math';
import { Annotation, Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

export default function TwoScalesOneTarget() {
  const tokens = useResolvedTokens();
  const narrow: [number, number][] = Array.from({ length: 200 }, (_, i) => {
    const z = -4 + (8 * i) / 199;
    return [z, normalPdf(z, { mu: -2, sigma2: 0.09 })];
  });
  const wide: [number, number][] = Array.from({ length: 200 }, (_, i) => {
    const z = -4 + (8 * i) / 199;
    return [z, normalPdf(z, { mu: 2, sigma2: 2.25 })];
  });

  return (
    <div className="widget-grid">
      <Plot width={420} height={220} xDomain={[-4, 4]} yDomain={[0, 1.4]} label="Two targets with very different local widths">
        <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
        <Curve points={narrow} color={tokens.series[0]!} width={2.5} />
        <Curve points={wide} color={tokens.series[1]!} width={2.5} />
        <Annotation x={-2} y={1.33} text="sd 0.3" />
        <Annotation x={2} y={0.28} text="sd 1.5" />
      </Plot>
      <p className="widget-readout">
        One fixed step size cannot be right for both: a step tuned to the narrow curve barely moves across the wide
        one, and a step tuned to the wide curve mostly misses the narrow one entirely.
      </p>
    </div>
  );
}
