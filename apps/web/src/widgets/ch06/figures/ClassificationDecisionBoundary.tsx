import { evalGrid, fitGPClassificationLaplace, gpClassificationPredict, linspace, pcg32, rbfKernel, standardNormal } from '@prml/math';
import { Axes, ContourField, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(7);
const PER_CLASS = 12;
function blob(cx: number, cy: number) {
  return Array.from({ length: PER_CLASS }, () => [cx + 0.7 * standardNormal(rng), cy + 0.7 * standardNormal(rng)]);
}
const CLASS0 = blob(-1.2, -1.2);
const CLASS1 = blob(1.2, 1.2);
const POINTS = [...CLASS0, ...CLASS1];
const TARGETS = [...CLASS0.map(() => 0), ...CLASS1.map(() => 1)];

const KERNEL = rbfKernel(1.2);
const MODEL = fitGPClassificationLaplace(KERNEL, POINTS, TARGETS, 1e-6);

const AXIS = linspace(-3.5, 3.5, 45);
const FIELD = evalGrid(AXIS, AXIS, (x, y) => gpClassificationPredict(MODEL, [x, y]).probability);

export default function ClassificationDecisionBoundary() {
  const tokens = useResolvedTokens();
  const fill = sequentialScale([0, 1]);

  return (
    <div className="widget-grid">
      <Plot height={320} xDomain={[-3.5, 3.5]} yDomain={[-3.5, 3.5]} equalAspect label="GP classification predicted probability, with the two training classes and the 0.5 decision boundary">
        <Heatmap data={FIELD} interpolator={fill} />
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
        <ContourField data={FIELD} levels={[0.5]} color={tokens.color.ink} lineWidth={2} />
        <ScatterField points={CLASS0.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `0-${i}` }))} color={tokens.series[0]!} size={5} />
        <ScatterField points={CLASS1.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `1-${i}` }))} color={tokens.series[1]!} size={5} />
      </Plot>
      <p className="widget-readout">
        {`${PER_CLASS} points per class, an RBF kernel, the Laplace approximation to the posterior over the latent function. The black line is where the predicted probability crosses 0.5, converged in ${MODEL.iterations} Newton steps.`}
      </p>
    </div>
  );
}
