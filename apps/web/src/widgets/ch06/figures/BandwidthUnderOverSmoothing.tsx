import { linspace, nadarayaWatsonPredict, pcg32, rbfKernel, standardNormal } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 14;
const GRID = linspace(0, 1, 140);
const SMALL_H = 0.015;
const LARGE_H = 0.5;

const rng = pcg32(20260615);
const XS = Array.from({ length: N }, () => rng.next()).sort((a, b) => a - b);
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + 0.2 * standardNormal(rng));
const TRAIN_VECS = XS.map((x) => [x]);

function curveAndMse(bandwidth: number) {
  const kernel = rbfKernel(bandwidth);
  const curve = GRID.map((x) => [x, nadarayaWatsonPredict(kernel, TRAIN_VECS, TS, [x])] as const);
  const truth = GRID.map((x) => Math.sin(2 * Math.PI * x));
  const mse = curve.reduce((s, [, y], i) => s + (y - truth[i]!) ** 2, 0) / GRID.length;
  return { curve, mse };
}

const SMALL = curveAndMse(SMALL_H);
const LARGE = curveAndMse(LARGE_H);

export default function BandwidthUnderOverSmoothing() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label="Nadaraya-Watson curves for a small and a large bandwidth">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={SMALL.curve} color={tokens.series[0]!} width={2} />
        <Curve points={LARGE.curve} color={tokens.series[1]!} width={2} />
        <ScatterField points={XS.map((x, i) => ({ x, y: TS[i]!, id: i }))} color={tokens.color.ink} size={3.5} />
        <Legend
          entries={[
            { label: `h = ${SMALL_H}`, color: tokens.series[0]!, mark: 'line' },
            { label: `h = ${LARGE_H}`, color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`Against the true sin(2*pi*x): mean squared error is ${SMALL.mse.toFixed(4)} at h = ${SMALL_H} and ${LARGE.mse.toFixed(4)} at h = ${LARGE_H}. Too small and the curve chases noise between points; too large and it flattens toward the average, missing the sine's own curvature.`}
      </p>
    </div>
  );
}
