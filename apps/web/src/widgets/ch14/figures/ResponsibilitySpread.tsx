import { Annotation, Axes, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { normalPdf } from '@prml/math';
import { BMA_MODELS, bmaMixtureData } from '../data.js';
import '../../widgets.css';

const N = 20;
const DATA = bmaMixtureData(N, 0.5);

function responsibility0(x: number): number {
  const w0 = 0.5 * normalPdf(x, BMA_MODELS[0]);
  const w1 = 0.5 * normalPdf(x, BMA_MODELS[1]);
  return w0 / (w0 + w1);
}

const RESP = DATA.map((d) => responsibility0(d.x));
const MEAN_RESP = RESP.reduce((s, r) => s + r, 0) / RESP.length;

export default function ResponsibilitySpread() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[-5, 5]} yDomain={[0, 1]} label="Responsibility of component 0 for each of 20 points from a real mixture">
      <Axes x={{ label: 'x' }} y={{ label: 'responsibility of component 0' }} grid />
      <Rule y={MEAN_RESP} label={`mean ${MEAN_RESP.toFixed(2)}`} />
      <ScatterField points={DATA.map((d, i) => ({ x: d.x, y: RESP[i]!, color: tokens.series[d.source]!, size: 4 }))} />
      <Annotation x={0} y={1} text="true mixing proportion is 0.5" dy={-8} />
    </Plot>
  );
}
