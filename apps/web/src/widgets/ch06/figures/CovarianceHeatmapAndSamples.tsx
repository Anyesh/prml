import { compositeKernel, gramMatrix, gpPriorSample, linspace, pcg32 } from '@prml/math';
import { Axes, Curve, Heatmap, Plot, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const GRID = linspace(0, 1, 40);
const GRID_VECS = GRID.map((x) => [x]);
const KERNEL = compositeKernel({ theta0: 1, theta1: 25, theta2: 0, theta3: 0 });
const GRAM = gramMatrix(KERNEL, GRID_VECS);

const rng = pcg32(1729);
const SAMPLES = Array.from({ length: 4 }, (_, i) => gpPriorSample(rng.fork(i + 1), KERNEL, GRID_VECS));

export default function CovarianceHeatmapAndSamples() {
  const tokens = useResolvedTokens();
  const scale = sequentialScale([0, 1]);

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, GRID.length - 1]} yDomain={[0, GRID.length - 1]} equalAspect label="Covariance matrix K over 40 points on [0, 1]">
        <Heatmap data={{ xs: GRAM.map((_, i) => i), ys: GRAM.map((_, i) => i), values: GRAM }} interpolator={scale} />
      </Plot>
      <Plot height={220} xDomain={[0, 1]} yDomain={[-2.5, 2.5]} label="Four functions drawn from exactly that covariance">
        <Axes x={{ label: 'x' }} y={{ label: 'y' }} grid zeroLine />
        {SAMPLES.map((curve, i) => (
          <Curve key={i} points={GRID.map((x, j) => [x, curve[j]!] as const)} color={tokens.series[i % tokens.series.length]!} width={1.5} />
        ))}
      </Plot>
      <p className="widget-readout">
        The bright diagonal band is where two points sit close enough for the kernel to
        correlate them strongly; its width is set by theta1. Every sampled function on the
        right stays smooth over exactly that width and decorrelates beyond it.
      </p>
    </div>
  );
}
