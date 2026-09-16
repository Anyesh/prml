import { useMemo, useState } from 'react';
import { bernoulliLogPmf, bernoulliSample, pcg32 } from '@prml/math';
import { NetworkDiagram, Plot } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { pCloudy as pC, pRain as pR, pSprinkler as pS, pWetGrass as pW } from './sprinklerNetwork.js';
import '../widgets.css';

type NodeId = 'C' | 'S' | 'R' | 'W';

const POSITIONS: Record<NodeId, { x: number; y: number }> = {
  C: { x: 2, y: 2.7 },
  S: { x: 0.6, y: 1.4 },
  R: { x: 3.4, y: 1.4 },
  W: { x: 2, y: 0.2 },
};

const EDGES: readonly [NodeId, NodeId][] = [
  ['C', 'S'],
  ['C', 'R'],
  ['S', 'W'],
  ['R', 'W'],
];

const NAMES: Record<NodeId, string> = { C: 'Cloudy', S: 'Sprinkler', R: 'Rain', W: 'Wet grass' };
const ORDER: readonly NodeId[] = ['C', 'S', 'R', 'W'];
const BASE_SEED = 20260913;

interface DrawStep {
  readonly node: NodeId;
  readonly value: 0 | 1;
  readonly conditional: number;
  readonly logJoint: number;
}

function drawTrace(seedTick: number): DrawStep[] {
  const rng = pcg32(BASE_SEED, seedTick + 1);
  const c = bernoulliSample(rng, { mu: pC() });
  const s = bernoulliSample(rng, { mu: pS(c) });
  const r = bernoulliSample(rng, { mu: pR(c) });
  const w = bernoulliSample(rng, { mu: pW(s, r) });

  const values: Record<NodeId, 0 | 1> = { C: c, S: s, R: r, W: w };
  const conditionals: Record<NodeId, number> = { C: pC(), S: pS(c), R: pR(c), W: pW(s, r) };

  let logJoint = 0;
  return ORDER.map((node) => {
    logJoint += bernoulliLogPmf(values[node], { mu: conditionals[node] });
    return { node, value: values[node], conditional: conditionals[node], logJoint };
  });
}

export default function AncestralSampling() {
  const [seedTick, setSeedTick] = useState(0);
  const [step, setStep] = useState(0);

  const trace = useMemo(() => drawTrace(seedTick), [seedTick]);
  const clampedStep = Math.min(step, trace.length - 1);
  const drawnSoFar = new Set(trace.slice(0, clampedStep + 1).map((t) => t.node));

  const nodes = ORDER.map((id) => {
    const pos = POSITIONS[id];
    const drawn = trace.find((t) => t.node === id);
    const shown = drawnSoFar.has(id);
    return {
      id,
      x: pos.x,
      y: pos.y,
      label: shown ? `${id}=${drawn!.value}` : id,
      observed: shown,
      dimmed: !shown,
    };
  });

  const edges = EDGES.map(([from, to]) => ({
    from,
    to,
    directed: true,
    dimmed: !(drawnSoFar.has(from) && drawnSoFar.has(to)),
  }));

  const current = trace[clampedStep]!;
  const labels = trace.map((t, i) => `${i === 0 ? 'p(' : '× p('}${t.node}${t.node === 'W' ? '|S,R' : t.node === 'C' ? '' : '|C'})`);

  return (
    <div>
      <div className="widget-grid">
        <Plot width={340} height={280} xDomain={[0, 4]} yDomain={[0, 3]} equalAspect label="The sprinkler network, nodes revealed in sampling order">
          <NetworkDiagram nodes={nodes} edges={edges} directed nodeLabel={(n) => NAMES[n.id as NodeId]} />
        </Plot>
        <div>
          <p className="widget-readout">
            {NAMES[current.node]} drawn as {current.value}, from a conditional of {current.conditional.toFixed(2)}.
          </p>
          <p className="widget-readout">
            Running joint so far: {trace
              .slice(0, clampedStep + 1)
              .map((t) => `p(${t.node}=${t.value}${t.node === 'C' ? '' : '|...'})`)
              .join(' · ')}{' '}
            = {Math.exp(current.logJoint).toFixed(4)}
          </p>
        </div>
      </div>
      <StepThrough step={clampedStep} stepCount={trace.length} onStep={setStep} labels={labels} />
      <button
        type="button"
        className="widget-reset"
        onClick={() => {
          setSeedTick((t) => t + 1);
          setStep(0);
        }}
      >
        New sample
      </button>
    </div>
  );
}
