import { Annotation, Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { normalPdf } from '@prml/math';
import { BMA_MODELS, bmaMixtureData } from '../data.js';
import '../../widgets.css';

const FULL = bmaMixtureData(20, 0.5);

function responsibility0(x: number): number {
  const w0 = 0.5 * normalPdf(x, BMA_MODELS[0]);
  const w1 = 0.5 * normalPdf(x, BMA_MODELS[1]);
  return w0 / (w0 + w1);
}

function meanResponsibility(n: number): number {
  const slice = FULL.slice(0, n);
  return slice.reduce((s, d) => s + responsibility0(d.x), 0) / slice.length;
}

const SIZES = [5, 20];
const MEANS = SIZES.map(meanResponsibility);

export default function ResponsibilityStability() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={180} xDomain={[0, 3]} yDomain={[0, 1]} label="Mean responsibility of component 0 at two dataset sizes">
      <Axes x={{ ticks: [1, 2], format: (v) => (v === 1 ? 'n = 5' : 'n = 20') }} y={{ label: 'mean responsibility' }} grid />
      <Bars bars={SIZES.map((_, i) => ({ at: i + 1, value: MEANS[i]!, color: tokens.color.accent }))} thickness={0.5} />
      <Annotation x={1} y={MEANS[0]!} text={MEANS[0]!.toFixed(3)} dy={-10} />
      <Annotation x={2} y={MEANS[1]!} text={MEANS[1]!.toFixed(3)} dy={-10} />
    </Plot>
  );
}
