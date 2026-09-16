import {
  backpropGradientBatch,
  flattenWeights,
  forwardPass,
  initializeWeights,
  mean,
  pcg32,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
  type NetworkWeights,
} from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [1, 10, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
const SHIFT = 0.1;
const LEARNING_RATE = 0.4;
const STEPS = 800;
const CENTRES = [-0.6, -0.2, 0.2, 0.6];

function boxTarget(x: number): number {
  return 0.5 * (Math.tanh(5 * (x + 0.5)) - Math.tanh(5 * (x - 0.5)));
}

function baselineDataset(): Dataset {
  return { inputs: CENTRES.map((x) => [x]), targets: CENTRES.map((x) => [boxTarget(x)]) };
}

function augmentedDataset(): Dataset {
  const inputs: number[][] = [];
  const targets: number[][] = [];
  for (const x of CENTRES) {
    for (const dx of [-SHIFT, 0, SHIFT]) {
      inputs.push([x + dx]);
      targets.push([boxTarget(x)]);
    }
  }
  return { inputs, targets };
}

function train(dataset: Dataset, seed: number): number[] {
  let flat = flattenWeights(initializeWeights(SPEC, pcg32(seed), 0.7));
  for (let step = 0; step < STEPS; step++) {
    const weights = unflattenWeights(SPEC, flat);
    const grad = flattenWeights(backpropGradientBatch(SPEC, weights, dataset, 'sumSquared'));
    flat = flat.map((w, i) => w - (LEARNING_RATE / dataset.inputs.length) * grad[i]!);
  }
  return flat;
}

function sensitivityAt(weights: NetworkWeights, x: number): number {
  const plus = forwardPass(SPEC, weights, [x + SHIFT]).output[0]!;
  const minus = forwardPass(SPEC, weights, [x - SHIFT]).output[0]!;
  return Math.abs((plus - minus) / (2 * SHIFT));
}

const BASELINE_WEIGHTS = unflattenWeights(SPEC, train(baselineDataset(), 20260610));
const AUGMENTED_WEIGHTS = unflattenWeights(SPEC, train(augmentedDataset(), 20260610));
const BASELINE_SENS = mean(CENTRES.map((x) => sensitivityAt(BASELINE_WEIGHTS, x)));
const AUGMENTED_SENS = mean(CENTRES.map((x) => sensitivityAt(AUGMENTED_WEIGHTS, x)));

export default function AugmentationReducesSensitivity() {
  const tokens = useResolvedTokens();
  const bound = Math.max(BASELINE_SENS, AUGMENTED_SENS) * 1.2;
  const bars = [
    { at: 0, value: BASELINE_SENS, color: tokens.color.danger },
    { at: 1, value: AUGMENTED_SENS, color: tokens.color.accent },
  ];

  return (
    <Plot height={220} xDomain={[-0.5, 1.5]} yDomain={[0, bound]} label="Average output sensitivity to a 0.1-wide shift, baseline against shift-augmented training">
      <Axes x={{ label: 'baseline (0) vs shift-augmented (1)' }} y={{ label: 'mean |Δy / Δx|' }} grid />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
