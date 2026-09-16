import { useMemo } from 'react';
import { smoFitClassifier, linearKernel } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LINEAR_KERNEL = linearKernel();

const POS: readonly (readonly [number, number])[] = [
  [1.0, 1.2], [2.1, 1.8], [1.6, -0.3], [2.8, 1.1], [0.6, -1.4], [-0.4, 0.6],
];
const NEG: readonly (readonly [number, number])[] = [
  [-0.8, -1.1], [0.3, 0.2], [1.1, -0.2], [-1.9, 0.4], [-0.6, 1.3], [0.9, 1.0],
];
const C_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 8, 12, 20, 40];

export default function SupportVectorCountVsC() {
  const tokens = useResolvedTokens();

  const counts = useMemo(() => {
    const points = [...POS, ...NEG];
    const labels = [...POS.map(() => 1 as const), ...NEG.map(() => -1 as const)];
    return C_GRID.map((C) => {
      const fit = smoFitClassifier(points, labels, LINEAR_KERNEL, { C });
      return { logC: Math.log10(C), count: fit.supportVectors.length };
    });
  }, []);

  const maxCount = Math.max(...counts.map((c) => c.count));

  return (
    <Plot height={220} xDomain={[-1.4, 1.7]} yDomain={[0, maxCount + 1]} label="Number of support vectors against C, log-scaled on the horizontal axis">
      <Axes x={{ label: 'log₁₀ C' }} y={{ label: 'support vectors' }} grid />
      <Curve points={counts.map((c) => [c.logC, c.count] as const)} color={tokens.color.accent} width={2} />
      <ScatterField points={counts.map((c, i) => ({ x: c.logC, y: c.count, id: i }))} color={tokens.color.accent} size={3.5} />
    </Plot>
  );
}
