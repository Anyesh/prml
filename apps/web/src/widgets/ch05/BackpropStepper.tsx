import { useState } from 'react';
import { backpropTrace, forwardPass, pcg32, initializeWeights, type NetworkSpec } from '@prml/math';
import { Annotation, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [2, 2, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
const WEIGHTS = initializeWeights(SPEC, pcg32(20260530), 1.3);
const INPUT = [0.5, -0.3];
const TARGET = [0.8];

const TRACE = backpropTrace(SPEC, WEIGHTS, INPUT, TARGET, 'sumSquared');
const FORWARD = forwardPass(SPEC, WEIGHTS, INPUT);

interface Event {
  readonly phase: 'forward' | 'backward';
  readonly layer: number;
  readonly unit: number;
  readonly label: string;
  readonly value: number;
}

const EVENTS: Event[] = [];
for (let l = 1; l <= 2; l++) {
  TRACE.activations[l]!.forEach((z, j) => EVENTS.push({ phase: 'forward', layer: l, unit: j, label: 'z', value: z }));
}
for (let l = 2; l >= 1; l--) {
  TRACE.deltas[l]!.forEach((d, j) => EVENTS.push({ phase: 'backward', layer: l, unit: j, label: 'δ', value: d }));
}

const LAYER_X = [0, 1, 2];
const LAYER_UNIT_COUNT = SPEC.layerSizes;

function unitY(layer: number, unit: number): number {
  const count = LAYER_UNIT_COUNT[layer]!;
  return unit - (count - 1) / 2;
}

export default function BackpropStepper() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();
  const revealed = EVENTS.slice(0, step);

  const inputPoints = INPUT.map((x, i) => ({ x: LAYER_X[0]!, y: unitY(0, i), id: `in-${i}`, value: x }));

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[-0.5, 2.5]} yDomain={[-2, 2]} label="Forward activations then backward deltas, revealed one unit at a time">
        <ScatterField points={inputPoints.map((p) => ({ x: p.x, y: p.y, id: p.id, color: tokens.color.ink, size: 6 }))} />
        {inputPoints.map((p) => (
          <Annotation key={p.id} x={p.x} y={p.y} dy={-14} text={`x=${p.value.toFixed(2)}`} color={tokens.color.inkMuted} size="xs" />
        ))}
        {revealed.map((e, i) => {
          const x = LAYER_X[e.layer]!;
          const y = unitY(e.layer, e.unit);
          const color = e.phase === 'forward' ? tokens.color.accent : tokens.color.danger;
          return (
            <ScatterField key={i} points={[{ x, y, id: `${e.phase}-${e.layer}-${e.unit}`, color, size: 7 }]} />
          );
        })}
        {revealed.map((e, i) => {
          const x = LAYER_X[e.layer]!;
          const y = unitY(e.layer, e.unit);
          const dy = e.phase === 'forward' ? -14 : 16;
          const color = e.phase === 'forward' ? tokens.color.accent : tokens.color.danger;
          return <Annotation key={`label-${i}`} x={x} y={y} dy={dy} text={`${e.label}=${e.value.toFixed(3)}`} color={color} size="xs" plate />;
        })}
      </Plot>
      <StepThrough step={step} stepCount={EVENTS.length + 1} onStep={setStep} interval={700} />
      <p className="widget-readout">
        {step === 0
          ? `Inputs x = (${INPUT[0]}, ${INPUT[1]}), target t = ${TARGET[0]}. Step forward to compute each unit's activation left to right, then backward to watch each δ arrive right to left.`
          : step <= 3
            ? `Forward pass: unit computes a weighted sum of the layer before it, then applies its activation. Output y = ${FORWARD.output[0]!.toFixed(3)} once all three are revealed.`
            : `Backward pass: δ starts at the output as y − t = ${(FORWARD.output[0]! - TARGET[0]!).toFixed(3)}, then each hidden δ is its own tanh derivative times the weighted sum of the δs feeding back from above.`}
      </p>
    </div>
  );
}
