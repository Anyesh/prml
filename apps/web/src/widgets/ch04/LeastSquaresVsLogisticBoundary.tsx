import { useMemo, useState } from 'react';
import { irlsFit, leastSquaresClassifierWeights, linspace, pcg32, standardNormal } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../widgets.css';

const SEED = 20260410;
const N_CORE = 25;
const N_EXTRA = 15;
const X_DOMAIN: readonly [number, number] = [-6, 6];
const Y_DOMAIN: readonly [number, number] = [-6, 10];
const LINE_X = linspace(-6, 6, 2);

function gaussianCluster(cx: number, cy: number, n: number, seed: number): { x: number; y: number }[] {
  const rng = pcg32(seed);
  return Array.from({ length: n }, () => ({
    x: cx + 0.7 * standardNormal(rng),
    y: cy + 0.7 * standardNormal(rng),
  }));
}

const CLASS_A = gaussianCluster(-2, 0, N_CORE, SEED);
const CLASS_B_CORE = gaussianCluster(2, 0, N_CORE, SEED + 1);
const EXTRA_JITTER = gaussianCluster(0, 0, N_EXTRA, SEED + 2).map((p) => [p.x * 0.5, p.y * 0.5] as const);

function boundaryLine(w: readonly number[]): readonly [number, number][] {
  const [w0, w1, w2] = w;
  if (w2 === undefined || Math.abs(w2) < 1e-9) return [];
  return LINE_X.map((x) => [x, -(w0! + w1! * x) / w2] as const);
}

export default function LeastSquaresVsLogisticBoundary() {
  const [extraY, setExtraY] = useState(0);
  const tokens = useResolvedTokens();

  const extraCluster = useMemo(
    () => EXTRA_JITTER.map(([dx, dy]) => ({ x: 2 + dx, y: extraY + dy })),
    [extraY],
  );
  const classB = useMemo(() => [...CLASS_B_CORE, ...extraCluster], [extraCluster]);

  const { lsLine, logisticLine } = useMemo(() => {
    const points = [...CLASS_A, ...classB];
    const design = points.map((p) => [1, p.x, p.y]);
    const targets1of2 = points.map((_, i) => (i < CLASS_A.length ? [1, 0] : [0, 1]));
    const wLs = leastSquaresClassifierWeights(design, targets1of2);
    const scoreDiff = [wLs[0]![0]! - wLs[0]![1]!, wLs[1]![0]! - wLs[1]![1]!, wLs[2]![0]! - wLs[2]![1]!];

    const targetsBinary = points.map((_, i) => (i < CLASS_A.length ? 0 : 1));
    const logistic = irlsFit(design, targetsBinary);

    return { lsLine: boundaryLine(scoreDiff), logisticLine: boundaryLine(logistic.weights) };
  }, [classB]);

  const markerId = 'extra-centroid';
  const markerPoint = { x: 2, y: extraY, id: markerId, shape: 'cross' as const, size: 7, color: tokens.color.danger };

  return (
    <div className="widget-grid">
      <Plot height={340} xDomain={X_DOMAIN} yDomain={Y_DOMAIN} label="Two classes with a draggable extra cluster">
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid zeroLine />
        {lsLine.length > 0 && <Curve points={lsLine} color={tokens.color.danger} width={2} />}
        {logisticLine.length > 0 && <Curve points={logisticLine} color={tokens.color.accent} width={2} dash="dashed" />}
        <ScatterField points={CLASS_A.map((p, i) => ({ ...p, id: `a${i}` }))} color={tokens.series[0]} size={3.5} />
        <ScatterField points={classB.map((p, i) => ({ ...p, id: `b${i}` }))} color={tokens.series[1]} size={3.5} />
        <ScatterField
          points={[markerPoint]}
          onMove={(_, _x, y) => setExtraY(y)}
          label={() => 'Drag to move the extra cluster of class-B points'}
        />
        <Legend
          entries={[
            { label: 'Least squares', color: tokens.color.danger, mark: 'line' },
            { label: 'Logistic regression', color: tokens.color.accent, mark: 'dashed-line' },
            { label: 'Drag the cross', color: tokens.color.danger, mark: 'dot' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {Math.abs(extraY) < 0.5
          ? 'The extra cluster sits on top of the core class-B cluster: both boundaries agree.'
          : `Extra cluster offset by ${extraY.toFixed(1)}. The least-squares boundary has rotated toward it even though every one of those points is already classified correctly; the logistic boundary has barely moved.`}
      </p>
    </div>
  );
}
