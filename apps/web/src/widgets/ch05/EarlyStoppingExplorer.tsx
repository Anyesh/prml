import { useMemo, useState } from 'react';
import {
  backpropGradientBatch,
  flattenWeights,
  initializeWeights,
  networkError,
  pcg32,
  standardNormal,
  unflattenWeights,
  type Dataset,
  type NetworkSpec,
} from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [1, 10, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
const N_TRAIN = 12;
const N_VAL = 12;
const LEARNING_RATE = 0.4;
const MAX_EPOCHS = 400;

function buildData(seed: number, n: number): Dataset {
  const rng = pcg32(seed);
  const inputs: number[][] = [];
  const targets: number[][] = [];
  for (let i = 0; i < n; i++) {
    const x = -1 + (2 * i) / (n - 1);
    const t = Math.sin(2 * Math.PI * x) * 0.6 + 0.25 * standardNormal(rng);
    inputs.push([x]);
    targets.push([t]);
  }
  return { inputs, targets };
}

const TRAIN = buildData(20260601, N_TRAIN);
const VAL = buildData(20260602, N_VAL);

function trainingCurve(): { trainErr: number[]; valErr: number[] } {
  let flat = flattenWeights(initializeWeights(SPEC, pcg32(20260603), 0.7));
  const trainErr: number[] = [];
  const valErr: number[] = [];
  for (let epoch = 0; epoch <= MAX_EPOCHS; epoch++) {
    const weights = unflattenWeights(SPEC, flat);
    trainErr.push(networkError(SPEC, weights, TRAIN, 'sumSquared') / N_TRAIN);
    valErr.push(networkError(SPEC, weights, VAL, 'sumSquared') / N_VAL);
    if (epoch === MAX_EPOCHS) break;
    const grad = flattenWeights(backpropGradientBatch(SPEC, weights, TRAIN, 'sumSquared'));
    flat = flat.map((w, i) => w - (LEARNING_RATE / N_TRAIN) * grad[i]!);
  }
  return { trainErr, valErr };
}

const CURVE = trainingCurve();
const STOP_EPOCH = CURVE.valErr.indexOf(Math.min(...CURVE.valErr));

export default function EarlyStoppingExplorer() {
  const [epoch, setEpoch] = useState(MAX_EPOCHS);
  const tokens = useResolvedTokens();

  const maxY = useMemo(() => Math.max(...CURVE.trainErr, ...CURVE.valErr) * 1.05, []);
  const trainPoints = CURVE.trainErr.slice(0, epoch + 1).map((e, i) => [i, e] as const);
  const valPoints = CURVE.valErr.slice(0, epoch + 1).map((e, i) => [i, e] as const);

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, MAX_EPOCHS]} yDomain={[0, maxY]} label="Training and validation error against epoch, with the early-stopping point marked">
        <Axes x={{ label: 'epoch' }} y={{ label: 'mean squared error' }} grid />
        <Rule x={STOP_EPOCH} color={tokens.color.inkFaint} />
        <Curve points={trainPoints} color={tokens.color.accent} width={2} />
        <Curve points={valPoints} color={tokens.color.danger} width={2} />
      </Plot>
      <Slider
        label="Stop training at epoch"
        value={epoch}
        onChange={setEpoch}
        min={0}
        max={MAX_EPOCHS}
        step={1}
        hint="Training error keeps falling the whole way; validation error turns back up once the network starts fitting this dataset's particular noise."
      />
      <p className="widget-readout">
        {`Validation error is lowest at epoch ${STOP_EPOCH} (train = ${CURVE.trainErr[STOP_EPOCH]!.toFixed(4)}, val = ${CURVE.valErr[STOP_EPOCH]!.toFixed(4)}). At epoch ${MAX_EPOCHS}, train has fallen to ${CURVE.trainErr[MAX_EPOCHS]!.toFixed(4)} while val has risen back to ${CURVE.valErr[MAX_EPOCHS]!.toFixed(4)}.`}
      </p>
    </div>
  );
}
