import { useMemo, useState } from 'react';
import { normalLogPdf, normalPdf } from '@prml/math';
import { Axes, Curve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { BMA_MODELS, bmaMixtureData, bmaSingleModelData } from './data.js';
import '../widgets.css';

export const title = 'Two data sources, the same two candidate shapes';
export const caption =
  'Drag the slider to reveal more points, drawn once and fixed. Left is the posterior over which single model made all of them; right is each point\'s own responsibility, on data a real mixture generated.';
export const figure = '14.1';

const MAX_N = 40;
const TRUE_MODEL = 0;
const SINGLE_DATA = bmaSingleModelData(MAX_N, TRUE_MODEL);
const MIXTURE_DATA = bmaMixtureData(MAX_N, 0.5);

function posteriorModel0(points: readonly number[]): number {
  let logP0 = Math.log(0.5);
  let logP1 = Math.log(0.5);
  for (const x of points) {
    logP0 += normalLogPdf(x, BMA_MODELS[0]);
    logP1 += normalLogPdf(x, BMA_MODELS[1]);
  }
  const m = Math.max(logP0, logP1);
  const p0 = Math.exp(logP0 - m);
  const p1 = Math.exp(logP1 - m);
  return p0 / (p0 + p1);
}

function responsibility0(x: number): number {
  const w0 = 0.5 * normalPdf(x, BMA_MODELS[0]);
  const w1 = 0.5 * normalPdf(x, BMA_MODELS[1]);
  return w0 / (w0 + w1);
}

export default function BmaVsMixture() {
  const [n, setN] = useState(6);
  const tokens = useResolvedTokens();

  const bmaCurve = useMemo(
    () => Array.from({ length: n }, (_, i) => [i + 1, posteriorModel0(SINGLE_DATA.slice(0, i + 1))] as const),
    [n],
  );

  const mixtureResp = useMemo(() => MIXTURE_DATA.slice(0, n).map((d) => ({ ...d, r: responsibility0(d.x) })), [n]);
  const runningMean = useMemo(
    () =>
      Array.from({ length: n }, (_, i) => {
        const slice = mixtureResp.slice(0, i + 1);
        return [i + 1, slice.reduce((s, d) => s + d.r, 0) / slice.length] as const;
      }),
    [mixtureResp, n],
  );

  const finalPosterior = posteriorModel0(SINGLE_DATA.slice(0, n));

  return (
    <div>
      <div className="widget-grid">
        <Plot height={260} xDomain={[1, MAX_N]} yDomain={[0, 1]} label="Posterior probability of model 0 as more points arrive">
          <Axes x={{ label: 'points observed' }} y={{ label: 'p(model 0 given data)' }} grid />
          <Rule y={0.5} />
          <Curve points={bmaCurve} color={tokens.color.accent} width={2.5} />
        </Plot>
        <Plot height={260} xDomain={[1, MAX_N]} yDomain={[0, 1]} label="Mean responsibility of component 0, with each point's own responsibility">
          <Axes x={{ label: 'points observed' }} y={{ label: 'responsibility of component 0' }} grid />
          <Rule y={0.5} />
          <ScatterField
            points={mixtureResp.map((d, i) => ({ x: i + 1, y: d.r, color: tokens.series[d.source]!, size: 3, opacity: 0.6 }))}
          />
          <Curve points={runningMean} color={tokens.color.accent} width={2.5} />
        </Plot>
      </div>
      <Slider label="points revealed" value={n} onChange={setN} min={1} max={MAX_N} step={1} />
      <p className="widget-readout">
        The left curve is heading toward {finalPosterior > 0.5 ? '1' : '0'}, because one model really did generate
        every point. The right curve keeps hugging 0.5 no matter how far the slider goes: a mixture does not
        concentrate the way a posterior over models does.
      </p>
    </div>
  );
}
