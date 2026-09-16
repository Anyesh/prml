import { useMemo } from 'react';
import {
  flattenWeights,
  forwardPass,
  initializeWeights,
  linspace,
  pcg32,
  regularizedGradient,
  standardNormal,
  unflattenWeights,
  backpropGradientBatch,
  type Dataset,
  type NetworkSpec,
} from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [1, 10, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
const N = 12;
const LEARNING_RATE = 0.4;
const STEPS = 400;
const LAMBDAS = [0, 0.01, 0.3] as const;
const GRID = linspace(-1, 1, 81);

function buildData(): Dataset {
  const rng = pcg32(20260601);
  const inputs: number[][] = [];
  const targets: number[][] = [];
  for (let i = 0; i < N; i++) {
    const x = -1 + (2 * i) / (N - 1);
    const t = Math.sin(2 * Math.PI * x) * 0.6 + 0.25 * standardNormal(rng);
    inputs.push([x]);
    targets.push([t]);
  }
  return { inputs, targets };
}

const DATA = buildData();

function trainWithDecay(lambda: number): number[] {
  let flat = flattenWeights(initializeWeights(SPEC, pcg32(20260603), 0.7));
  for (let step = 0; step < STEPS; step++) {
    const weights = unflattenWeights(SPEC, flat);
    const raw = backpropGradientBatch(SPEC, weights, DATA, 'sumSquared');
    const grad = flattenWeights(regularizedGradient(SPEC, weights, raw, lambda * N));
    flat = flat.map((w, i) => w - (LEARNING_RATE / N) * grad[i]!);
  }
  return flat;
}

export default function WeightDecayShrinkage() {
  const tokens = useResolvedTokens();
  const curves = useMemo(
    () =>
      LAMBDAS.map((lambda) => {
        const weights = unflattenWeights(SPEC, trainWithDecay(lambda));
        return GRID.map((x) => [x, forwardPass(SPEC, weights, [x]).output[0]!] as const);
      }),
    [],
  );

  return (
    <Plot height={260} xDomain={[-1, 1]} yDomain={[-1.4, 1.4]} label="Fits at three weight-decay strengths on the same noisy data">
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      {curves.map((points, i) => (
        <Curve key={i} points={points} color={tokens.series[i % tokens.series.length]!} width={2} />
      ))}
      <ScatterField points={DATA.inputs.map((x, i) => ({ x: x[0]!, y: DATA.targets[i]![0]!, id: i }))} color={tokens.color.ink} size={3.5} />
    </Plot>
  );
}
