import { crossEntropyError, linspace, sigmoid } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

// One outlier (x = -3, t = 1) sitting deep in class 0's territory: the sigmoid saturates
// on the wrong side of it, and that is exactly where the two error surfaces diverge.
const XS = [-2.2, -1.9, -1.6, 1.6, 1.9, 2.2, -3.0];
const TS = [0, 0, 0, 1, 1, 1, 1];
const DESIGN = XS.map((x) => [x]);
const W_GRID = linspace(-3, 3, 300);

function squaredErrorOnSigmoid(w: number): number {
  let sum = 0;
  for (let i = 0; i < XS.length; i++) {
    const y = sigmoid(w * XS[i]!);
    sum += (y - TS[i]!) ** 2;
  }
  return sum;
}

export default function ConvexVsNonconvexErrorSurface() {
  const tokens = useResolvedTokens();
  const crossEntropy = W_GRID.map((w) => crossEntropyError(DESIGN, TS, [w]));
  const squared = W_GRID.map(squaredErrorOnSigmoid);
  const crossEntropyNorm = Math.max(...crossEntropy);
  const squaredNorm = Math.max(...squared);

  return (
    <Plot height={240} xDomain={[-3, 3]} yDomain={[0, 1.05]} label="Error against a single weight, both normalised to a peak of 1">
      <Axes x={{ label: 'w' }} y={{ label: 'E(w) / max E(w)' }} grid />
      <Curve points={W_GRID.map((w, i) => [w, crossEntropy[i]! / crossEntropyNorm] as const)} color={tokens.color.accent} width={2} />
      <Curve points={W_GRID.map((w, i) => [w, squared[i]! / squaredNorm] as const)} color={tokens.color.danger} width={2} />
      <Legend
        entries={[
          { label: 'cross-entropy: one bowl', color: tokens.color.accent, mark: 'line' },
          { label: 'squared error on σ(wx): flattens both ways', color: tokens.color.danger, mark: 'line' },
        ]}
        placement="top-right"
      />
    </Plot>
  );
}
