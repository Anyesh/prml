import { useMemo } from 'react';
import { linspace, rvmRegressionFit, standardNormal, pcg32 } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N = 11;
const GAMMA = 3;

function rbfRow(x: number, centers: readonly number[]): number[] {
  return [1, ...centers.map((c) => Math.exp(-GAMMA * (x - c) ** 2))];
}

export default function GammaPerWeightBars() {
  const tokens = useResolvedTokens();

  const fit = useMemo(() => {
    const rng = pcg32(20260916);
    const xs = linspace(-1, 1, N);
    const trueW = new Array(N + 1).fill(0);
    trueW[0] = 0.2;
    trueW[3] = 1.5;
    trueW[7] = -1.2;
    const design = xs.map((x) => rbfRow(x, xs));
    const ts = design.map((row) => row.reduce((s, v, i) => s + v * trueW[i]!, 0) + 0.05 * standardNormal(rng));
    return rvmRegressionFit(design, ts, new Array(design[0]!.length).fill(1), 1);
  }, []);

  return (
    <Plot height={200} xDomain={[-0.5, fit.gamma.length - 0.5]} yDomain={[0, 1]} label="Per-weight gamma_i after convergence, one bar per basis function">
      <Axes x={{ label: 'basis index (0 = bias)' }} y={{ label: 'γᵢ' }} grid />
      <Bars
        bars={fit.gamma.map((g, i) => ({ at: i, value: g, color: g > 0.05 ? tokens.color.accent : tokens.color.inkFaint }))}
        thickness={0.7}
      />
    </Plot>
  );
}
