import { useMemo, useState } from 'react';
import {
  designMatrix,
  dot,
  dualPredict,
  dualRidgeCoefficients,
  eye,
  gramMatrix,
  kernelFromFeatureMap,
  linspace,
  matAdd,
  matmul,
  pcg32,
  polynomialBasis,
  regularisedWeights,
  standardNormal,
  transpose,
} from '@prml/math';
import { Axes, Curve, Heatmap, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const N = 6;
const GRID = linspace(0, 1, 120);

function buildDataset() {
  const rng = pcg32(20260601);
  const xs = Array.from({ length: N }, () => rng.next()).sort((a, b) => a - b);
  const ts = xs.map((x) => Math.sin(2 * Math.PI * x) + 0.15 * standardNormal(rng));
  return { xs, ts };
}

const DATASET = buildDataset();

function toField(matrix: readonly (readonly number[])[]) {
  const idx = matrix.map((_, i) => i);
  return { xs: idx, ys: idx, values: matrix };
}

export default function DualVsPrimalRidge() {
  const [degree, setDegree] = useState(3);
  const [logLambda, setLogLambda] = useState(-2);
  const tokens = useResolvedTokens();
  const lambda = Math.pow(10, logLambda);

  const result = useMemo(() => {
    const phi = polynomialBasis(degree, { bias: true });
    const design = designMatrix(DATASET.xs, phi);
    const wPrimal = regularisedWeights(design, DATASET.ts, lambda);
    const m = wPrimal.length;
    const primalMatrix = matAdd(eye(m, lambda), matmul(transpose(design), design));

    const kernel = kernelFromFeatureMap((v: readonly number[]) => phi(v[0]!));
    const trainVecs = DATASET.xs.map((x) => [x]);
    const gram = gramMatrix(kernel, trainVecs);
    const aDual = dualRidgeCoefficients(gram, DATASET.ts, lambda);
    const dualMatrix = matAdd(gram, eye(N, lambda));

    const curve = GRID.map((x) => {
      const primal = dot(phi(x), wPrimal);
      const dual = dualPredict(kernel, trainVecs, aDual, [x]);
      return { x, primal, dual };
    });
    const maxDiff = curve.reduce((acc, p) => Math.max(acc, Math.abs(p.primal - p.dual)), 0);

    return { primalMatrix, dualMatrix, curve, maxDiff, m };
  }, [degree, lambda]);

  const primalScale = sequentialScale([0, Math.max(...result.primalMatrix.flat().map(Math.abs))]);
  const dualScale = sequentialScale([0, Math.max(...result.dualMatrix.flat().map(Math.abs))]);

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[0, 1]} yDomain={[-1.6, 1.6]} label="Weight-space and dual-space ridge predictions, overlaid">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={result.curve.map((p) => [p.x, p.primal] as const)} color={tokens.series[0]!} width={3} />
        <Curve points={result.curve.map((p) => [p.x, p.dual] as const)} color={tokens.series[1]!} width={1.25} dash="dashed" />
        <ScatterField points={DATASET.xs.map((x, i) => ({ x, y: DATASET.ts[i]!, id: i }))} color={tokens.color.ink} size={4} />
      </Plot>

      <Plot height={220} xDomain={[0, result.m - 1]} yDomain={[0, result.m - 1]} equalAspect label={`The ${result.m} by ${result.m} matrix inverted in weight space`}>
        <Heatmap data={toField(result.primalMatrix)} interpolator={primalScale} />
      </Plot>

      <Plot height={220} xDomain={[0, N - 1]} yDomain={[0, N - 1]} equalAspect label={`The ${N} by ${N} matrix inverted in dual space`}>
        <Heatmap data={toField(result.dualMatrix)} interpolator={dualScale} />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="Polynomial degree (M = degree + 1 features)" value={degree} onChange={setDegree} min={1} max={5} step={1} />
        <Slider label="log10 lambda" value={logLambda} onChange={setLogLambda} min={-4} max={1} step={0.1} />
        <p className="widget-readout">
          {`Weight space inverts a ${result.m} by ${result.m} matrix; dual space inverts ${N} by ${N}, always the number of training points. ` +
            `The two prediction curves differ by at most ${result.maxDiff.toExponential(2)} across the grid: the same ridge solution, computed two ways.`}
        </p>
      </Panel>
    </div>
  );
}
