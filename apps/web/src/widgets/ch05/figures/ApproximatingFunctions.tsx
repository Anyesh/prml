import { useMemo } from 'react';
import {
  backpropGradientBatch,
  flattenWeights,
  forwardPass,
  initializeWeights,
  linspace,
  pcg32,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
} from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [1, 3, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
const TRAIN_XS = linspace(-1, 1, 21);
const LEARNING_RATE = 0.5;
const STEPS = 3000;
const GRID = linspace(-1, 1, 81);

const TARGETS: readonly { readonly name: string; readonly fn: (x: number) => number }[] = [
  { name: 'x²', fn: (x) => x * x },
  { name: 'sin(3x)', fn: (x) => Math.sin(3 * x) },
  { name: '|x|', fn: (x) => Math.abs(x) },
  { name: 'step(x)', fn: (x) => (x >= 0 ? 1 : 0) },
];

function train(fn: (x: number) => number, seed: number): number[] {
  const dataset: Dataset = { inputs: TRAIN_XS.map((x) => [x]), targets: TRAIN_XS.map((x) => [fn(x)]) };
  let flat = flattenWeights(initializeWeights(SPEC, pcg32(seed), 1.0));
  for (let step = 0; step < STEPS; step++) {
    const weights = unflattenWeights(SPEC, flat);
    const grad = flattenWeights(backpropGradientBatch(SPEC, weights, dataset, 'sumSquared'));
    flat = flat.map((w, i) => w - (LEARNING_RATE / TRAIN_XS.length) * grad[i]!);
  }
  return flat;
}

export default function ApproximatingFunctions() {
  const tokens = useResolvedTokens();
  const fits = useMemo(
    () =>
      TARGETS.map(({ name, fn }, i) => {
        const weights = unflattenWeights(SPEC, train(fn, 20260510 + i));
        const curve = GRID.map((x) => [x, forwardPass(SPEC, weights, [x]).output[0]!] as const);
        const target = GRID.map((x) => [x, fn(x)] as const);
        return { name, curve, target };
      }),
    [],
  );

  return (
    <div className="widget-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}>
      {fits.map(({ name, curve, target }) => (
        <Plot key={name} height={150} xDomain={[-1, 1]} yDomain={[-1.3, 1.3]} label={`Three tanh hidden units fitting ${name}`}>
          <Axes x={{ label: 'x' }} y={{ label: name }} grid />
          <Curve points={target} color={tokens.color.inkFaint} width={1.5} dash="dashed" />
          <Curve points={curve} color={tokens.color.accent} width={2} />
        </Plot>
      ))}
    </div>
  );
}
