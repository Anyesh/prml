import { normalPdf, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N_POINTS = 400;

export default function SliceAugmentedArea() {
  const tokens = useResolvedTokens();
  const rng = pcg32(9, 110);
  const pdf = (z: number) => normalPdf(z, { mu: 0, sigma2: 1 });
  const curve: [number, number][] = Array.from({ length: 200 }, (_, i) => {
    const z = -4 + (8 * i) / 199;
    return [z, pdf(z)];
  });

  const points = Array.from({ length: N_POINTS }, () => {
    const z = -4 + 8 * rng.next();
    const u = rng.next() * 0.42;
    return { z, u, under: u <= pdf(z) };
  }).filter((p) => p.under);

  return (
    <div className="widget-grid">
      <Plot width={420} height={220} xDomain={[-4, 4]} yDomain={[0, 0.42]} label="Uniform points under the curve, the augmented (z, u) space slice sampling explores">
        <Axes x={{ label: 'z' }} y={{ label: 'u' }} grid />
        <Curve points={curve} color={tokens.series[0]!} width={2} />
        <ScatterField points={points.map((p) => ({ x: p.z, y: p.u, color: tokens.color.accentWash, size: 2 }))} />
      </Plot>
      <p className="widget-readout">
        Sampling z from p(z) is exactly sampling (z, u) uniformly under the curve and dropping u (11.51-11.52). Slice
        sampling never draws this whole cloud; it only ever looks at one horizontal slice of it at a time.
      </p>
    </div>
  );
}
