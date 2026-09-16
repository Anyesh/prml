import { useMemo, useState } from 'react';
import {
  evalGrid,
  fitSeparateCovarianceGaussian,
  fitSharedCovarianceGaussian,
  linspace,
  mvnSample,
  pcg32,
  posteriorSeparateCovariance,
  posteriorSharedCovariance,
} from '@prml/math';
import { Axes, ContourField, Curve, FunctionCurve, Legend, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { Toggle } from '@prml/ui';
import '../widgets.css';

const COV1 = [
  [1.4, 0.6],
  [0.6, 0.5],
];
const COV2 = [
  [0.5, -0.3],
  [-0.3, 1.0],
];
const N = 45;
const DOMAIN: readonly [number, number] = [-5, 5];
const GRID = linspace(-5, 5, 90);

export default function GaussianGenerativeExplorer() {
  const [mean1, setMean1] = useState<[number, number]>([-1.8, 0.5]);
  const [mean2, setMean2] = useState<[number, number]>([1.8, -0.5]);
  const [shared, setShared] = useState(true);
  const tokens = useResolvedTokens();

  const points = useMemo(() => {
    const rng = pcg32(20260420);
    const class1 = Array.from({ length: N }, () => mvnSample(rng, { mean: mean1, cov: COV1 }));
    const class2 = Array.from({ length: N }, () => mvnSample(rng, { mean: mean2, cov: COV2 }));
    return { class1, class2 };
  }, [mean1, mean2]);

  const posteriorAt = useMemo(() => {
    if (shared) {
      const fit = fitSharedCovarianceGaussian([points.class1, points.class2]);
      return (x: number, y: number) => posteriorSharedCovariance([x, y], fit)[0]!;
    }
    const fit = fitSeparateCovarianceGaussian([points.class1, points.class2]);
    return (x: number, y: number) => posteriorSeparateCovariance([x, y], fit)[0]!;
  }, [points, shared]);

  const posteriorField = useMemo(() => evalGrid(GRID, GRID, posteriorAt), [posteriorAt]);
  const fill = useMemo(() => sequentialScale([0, 1]), []);

  const midline = useMemo(() => {
    const mid: [number, number] = [(mean1[0]! + mean2[0]!) / 2, (mean1[1]! + mean2[1]!) / 2];
    const dx = mean2[0]! - mean1[0]!;
    const dy = mean2[1]! - mean1[1]!;
    const len = Math.hypot(dx, dy) || 1;
    return { mid, unit: [dx / len, dy / len] as const };
  }, [mean1, mean2]);

  const slicePosterior = useMemo(() => {
    return (t: number) => posteriorAt(midline.mid[0] + t * midline.unit[0], midline.mid[1] + t * midline.unit[1]);
  }, [posteriorAt, midline]);

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Class-conditional densities and the resulting posterior">
        <ContourField data={posteriorField} levelCount={9} color={tokens.color.inkFaint} fill={fill} z={-2} />
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
        <ScatterField
          points={[{ x: mean1[0], y: mean1[1], id: 'm1', shape: 'cross', size: 8, color: tokens.color.ink }]}
          onMove={(_, x, y) => setMean1([x, y])}
        />
        <ScatterField
          points={[{ x: mean2[0], y: mean2[1], id: 'm2', shape: 'cross', size: 8, color: tokens.color.ink }]}
          onMove={(_, x, y) => setMean2([x, y])}
        />
        <ScatterField points={points.class1.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `1-${i}` }))} color={tokens.series[0]} size={2.5} opacity={0.6} />
        <ScatterField points={points.class2.map((p, i) => ({ x: p[0]!, y: p[1]!, id: `2-${i}` }))} color={tokens.series[1]} size={2.5} opacity={0.6} />
      </Plot>

      <Plot height={220} xDomain={[-6, 6]} yDomain={[-0.05, 1.05]} label="Posterior p(C1|x) along the line joining the two means">
        <Axes x={{ label: 'position along the mean-difference line' }} y={{ label: 'p(C1)' }} grid />
        <Curve points={[[-6, 0.5], [6, 0.5]]} color={tokens.color.inkFaint} width={1} dash="dotted" />
        <FunctionCurve f={slicePosterior} domain={[-6, 6]} color={tokens.color.accent} width={2} />
        <Legend entries={[{ label: shared ? 'shared Σ: exact sigmoid of a linear function' : 'separate Σ: no longer a plain sigmoid slice', color: tokens.color.accent, mark: 'line' }]} placement="top-right" />
      </Plot>

      <Toggle
        label="Shared covariance"
        checked={shared}
        onChange={setShared}
        hint="On: one Σ for both classes, linear boundary. Off: each class keeps its own Σ, quadratic boundary."
      />
    </div>
  );
}
