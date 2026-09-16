import { Annotation, Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import { normalLogPdf } from '@prml/math';
import { BMA_MODELS, bmaSingleModelData } from '../data.js';
import '../../widgets.css';

const N = 20;
const DATA = bmaSingleModelData(N, 0);

function posterior(points: readonly number[]): readonly [number, number] {
  let log0 = Math.log(0.5);
  let log1 = Math.log(0.5);
  for (const x of points) {
    log0 += normalLogPdf(x, BMA_MODELS[0]);
    log1 += normalLogPdf(x, BMA_MODELS[1]);
  }
  const m = Math.max(log0, log1);
  const p0 = Math.exp(log0 - m);
  const p1 = Math.exp(log1 - m);
  return [p0 / (p0 + p1), p1 / (p0 + p1)];
}

const [post0, post1] = posterior(DATA);

export default function PosteriorOverModels() {
  const tokens = useResolvedTokens();
  const offset = 0.18;

  const bars = [
    { at: 0 - offset, value: 0.5, color: tokens.series[0]! },
    { at: 0 + offset, value: 0.5, color: tokens.series[1]! },
    { at: 1 - offset, value: post0, color: tokens.series[0]! },
    { at: 1 + offset, value: post1, color: tokens.series[1]! },
  ];

  return (
    <Plot height={200} xDomain={[-0.6, 1.6]} yDomain={[0, 1]} label="Prior versus posterior probability over the two candidate models, PRML equation 14.6">
      <Axes x={{ ticks: [0, 1], format: (v) => (v === 0 ? 'prior' : 'posterior') }} y={{ label: 'p(h | data)' }} grid />
      <Bars bars={bars} thickness={0.3} />
      <Annotation x={1 - offset} y={post0} text={post0.toFixed(3)} dy={-10} />
      <Annotation x={1 + offset} y={post1} text={post1.toFixed(3)} dy={-10} />
    </Plot>
  );
}
