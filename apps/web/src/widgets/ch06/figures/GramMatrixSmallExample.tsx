import { gramMatrix, rbfKernel } from '@prml/math';
import { Annotation, Axes, Heatmap, Plot, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const POINTS = [0.1, 0.35, 0.6, 0.9];
const LENGTH_SCALE = 0.3;
const KERNEL = rbfKernel(LENGTH_SCALE);
const GRAM = gramMatrix(KERNEL, POINTS.map((x) => [x]));
const N = POINTS.length;
const FIELD = { xs: POINTS.map((_, i) => i), ys: POINTS.map((_, i) => i), values: GRAM };

export default function GramMatrixSmallExample() {
  const tokens = useResolvedTokens();
  const scale = sequentialScale([0, 1]);

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-0.5, N - 0.5]} yDomain={[-0.5, N - 0.5]} equalAspect label={`The Gram matrix for ${N} points under an RBF kernel`}>
        <Heatmap data={FIELD} interpolator={scale} />
        <Axes x={{ label: 'training point n', ticks: FIELD.xs }} y={{ label: 'training point m', ticks: FIELD.ys }} />
        {GRAM.map((row, i) => row.map((v, j) => (
          <Annotation key={`${i}-${j}`} x={j} y={i} text={v.toFixed(2)} color={i === j ? tokens.color.bg : tokens.color.ink} plate />
        )))}
      </Plot>
      <p className="widget-readout">
        {`Four points, an RBF kernel of length scale ${LENGTH_SCALE}. Every entry is a number computed by a single call to k(xn, xm); the diagonal is always exactly 1, because k(x, x) does not depend on the length scale for this kernel.`}
      </p>
    </div>
  );
}
