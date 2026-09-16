import { useMemo } from 'react';
import { forwardPass, initializeWeights, linspace, pcg32, type NetworkSpec, type NetworkWeights } from '@prml/math';
import { Annotation, Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [1, 2, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
const GRID = linspace(-3, 3, 81);
const BASE = initializeWeights(SPEC, pcg32(20260520), 1.4);

function permuteAndFlip(weights: NetworkWeights): NetworkWeights {
  const hidden = weights[0]!;
  const output = weights[1]!;
  const swappedHidden = [hidden[1]!, hidden[0]!.map((w) => -w)];
  const swappedOutput = [[output[0]![0]!, output[0]![2]!, -output[0]![1]!]];
  return [swappedHidden, swappedOutput];
}

const TRANSFORMED = permuteAndFlip(BASE);

function curveOf(weights: NetworkWeights) {
  return GRID.map((x) => [x, forwardPass(SPEC, weights, [x]).output[0]!] as const);
}

export default function WeightSpaceSymmetry() {
  const tokens = useResolvedTokens();
  const { baseCurve, transformedCurve, maxGap } = useMemo(() => {
    const a = curveOf(BASE);
    const b = curveOf(TRANSFORMED);
    const gap = Math.max(...a.map(([, y], i) => Math.abs(y - b[i]![1])));
    return { baseCurve: a, transformedCurve: b, maxGap: gap };
  }, []);

  return (
    <Plot height={220} xDomain={[-3, 3]} yDomain={[-3, 3]} label="Two different weight vectors computing the same function">
      <Axes x={{ label: 'x' }} y={{ label: 'y(x, w)' }} grid zeroLine />
      <Curve points={baseCurve} color={tokens.color.ink} width={3} opacity={0.6} />
      <Curve points={transformedCurve} color={tokens.color.accent} width={1.5} dash="dashed" />
      <Annotation x={-2.9} y={2.6} anchor="start" size="xs" color={tokens.color.inkMuted} text={`max |Δy| over the grid: ${maxGap.toExponential(2)}`} plate />
    </Plot>
  );
}
