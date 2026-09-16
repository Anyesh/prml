import { useMemo, useState } from 'react';
import { marginalFromMessages, maxSum, sumProduct, type FactorGraph, type Message } from '@prml/math';
import { NetworkDiagram, Plot } from '@prml/viz';
import { Select, StepThrough } from '@prml/ui';
import '../widgets.css';

const GRAPH: FactorGraph = {
  variables: [
    { id: 'x1', states: 2 },
    { id: 'x2', states: 3 },
    { id: 'x3', states: 2 },
    { id: 'x4', states: 2 },
  ],
  factors: [
    { id: 'fa', scope: ['x1', 'x3'], table: [1.0, 0.4, 0.9, 1.7] },
    { id: 'fb', scope: ['x2', 'x3'], table: [0.5, 1.2, 1.0, 0.3, 2.1, 0.8] },
    { id: 'fc', scope: ['x3', 'x4'], table: [1.4, 0.6, 0.2, 1.1] },
  ],
};

const ROOT = 'x3';

const POSITIONS: Record<string, { x: number; y: number; shape: 'circle' | 'square' }> = {
  x1: { x: 0.3, y: 2.6, shape: 'circle' },
  fa: { x: 1.5, y: 2.2, shape: 'square' },
  x3: { x: 2.7, y: 1.6, shape: 'circle' },
  fb: { x: 1.5, y: 1.0, shape: 'square' },
  x2: { x: 0.3, y: 0.6, shape: 'circle' },
  fc: { x: 3.9, y: 1.6, shape: 'square' },
  x4: { x: 5.1, y: 1.6, shape: 'circle' },
};

const BASE_EDGES: readonly [string, string][] = [
  ['x1', 'fa'],
  ['fa', 'x3'],
  ['x2', 'fb'],
  ['fb', 'x3'],
  ['x3', 'fc'],
  ['fc', 'x4'],
];

function pairKey(a: string, b: string): string {
  return [a, b].sort().join('-');
}

function formatValues(values: readonly number[]): string {
  return `[${values.map((v) => v.toFixed(2)).join(', ')}]`;
}

function lastMessageOn(played: readonly Message[], a: string, b: string): Message | undefined {
  for (let i = played.length - 1; i >= 0; i--) {
    const m = played[i]!;
    if (pairKey(m.from, m.to) === pairKey(a, b)) return m;
  }
  return undefined;
}

const MODE_OPTIONS = [
  { value: 'sumProduct', label: 'Sum-product (marginals)' },
  { value: 'maxSum', label: 'Max-sum (MAP configuration)' },
];

export default function FactorGraphInference() {
  const [mode, setMode] = useState<'sumProduct' | 'maxSum'>('sumProduct');
  const [step, setStep] = useState(0);

  const sp = useMemo(() => sumProduct(GRAPH, ROOT), []);
  const ms = useMemo(() => maxSum(GRAPH, ROOT), []);

  const totalSteps = mode === 'sumProduct' ? sp.schedule.length + 1 : ms.schedule.length + ms.backtrack.length + 1;
  const clampedStep = Math.min(step, totalSteps - 1);

  const playedMessages: readonly Message[] =
    mode === 'sumProduct' ? sp.schedule.slice(0, clampedStep) : ms.schedule.slice(0, Math.min(clampedStep, ms.schedule.length));

  const backtrackRevealed = mode === 'maxSum' ? ms.backtrack.slice(0, Math.max(0, clampedStep - ms.schedule.length)) : [];
  const revealedValues = new Map(backtrackRevealed.map((b) => [b.node, b.value]));

  const nodes = Object.entries(POSITIONS).map(([id, pos]) => {
    const revealed = revealedValues.get(id);
    return {
      id,
      x: pos.x,
      y: pos.y,
      shape: pos.shape,
      label: revealed !== undefined ? `${id}=${revealed}` : id,
      observed: revealed !== undefined,
    };
  });

  const edges = BASE_EDGES.map(([a, b]) => {
    const message = lastMessageOn(playedMessages, a, b);
    if (!message) return { from: a, to: b, dimmed: true };
    return {
      from: message.from,
      to: message.to,
      directed: true,
      label: formatValues(message.values),
    };
  });

  const rootMarginal = marginalFromMessages(GRAPH, ROOT, playedMessages);
  const labels =
    mode === 'sumProduct'
      ? ['Nothing sent yet', ...sp.schedule.map((m, i) => `Message ${i + 1}: ${m.from} -> ${m.to}`)]
      : [
          'Nothing sent yet',
          ...ms.schedule.map((m, i) => `Message ${i + 1}: ${m.from} -> ${m.to}`),
          ...ms.backtrack.map((b) => `Backtrack: ${b.node} = ${b.value}`),
        ];

  return (
    <div>
      <Select label="Algorithm" value={mode} options={MODE_OPTIONS} onChange={(v) => { setMode(v as 'sumProduct' | 'maxSum'); setStep(0); }} />
      <Plot width={520} height={280} xDomain={[0, 5.5]} yDomain={[0, 3]} equalAspect label="Circles are variables, squares are factors">
        <NetworkDiagram nodes={nodes} edges={edges} />
      </Plot>
      <StepThrough step={clampedStep} stepCount={totalSteps} onStep={setStep} labels={labels} />
      {mode === 'sumProduct' ? (
        <p className="widget-readout">
          Marginal at x3 from messages received so far: {formatValues(rootMarginal)}. It only becomes exact once every
          message has arrived.
        </p>
      ) : (
        <p className="widget-readout">
          {clampedStep <= ms.schedule.length
            ? 'Collecting log-messages towards x3. Once all have arrived, x3 picks its best value and the backtrack begins.'
            : `Backtracking: ${[...revealedValues.entries()].map(([n, v]) => `${n}=${v}`).join(', ')}.`}
          {clampedStep === totalSteps - 1
            ? ` Full MAP configuration recovered, log-probability ${ms.maxLogValue.toFixed(3)}.`
            : ''}
        </p>
      )}
    </div>
  );
}
