import { useMemo, useState } from 'react';
import {
  hmmBackwardScaled,
  hmmForwardScaled,
  hmmGammaScaled,
  hmmGaussianEmissionMatrix,
  hmmViterbi,
} from '@prml/math';
import { NetworkDiagram, Plot, layeredLayout, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { hmmDemo } from './data.js';

import '../widgets.css';

export const title = 'The trellis: Viterbi against the marginals';
export const caption =
  'Step forward to watch the surviving edge into every state light up column by column, then step once more to see the single best path picked out and compared against the largest node in each column.';
export const figure = '13.16';

const DEMO = hmmDemo();
const K = DEMO.components.length;
const N = DEMO.data.length;

export default function TrellisViterbi() {
  const tokens = useResolvedTokens();
  const [step, setStep] = useState(0);

  const { gamma, viterbi } = useMemo(() => {
    const B = hmmGaussianEmissionMatrix(DEMO.data, DEMO.components);
    const { alphaHat, c } = hmmForwardScaled(DEMO.pi, DEMO.A, B);
    const betaHat = hmmBackwardScaled(DEMO.A, B, c);
    const gammaOut = hmmGammaScaled(alphaHat, betaHat);
    const viterbiOut = hmmViterbi(DEMO.pi, DEMO.A, B);
    return { gamma: gammaOut, viterbi: viterbiOut };
  }, []);

  const sweepColumn = Math.min(step, N - 1);
  const backtracking = step >= N;

  const layout = useMemo(() => layeredLayout(new Array(N).fill(K), { x0: 0.06, x1: 0.94, y0: 0.1, y1: 0.9 }), []);
  const nodeAt = (t: number, k: number) => layout.find((n) => n.layer === t && n.index === k)!;

  const gammaArgmax = gamma.map((row) => row.indexOf(Math.max(...row)));

  const nodes = layout
    .filter((n) => n.layer <= sweepColumn)
    .map((n) => {
      const t = n.layer;
      const k = n.index;
      const onPath = backtracking && viterbi.path[t] === k;
      const isGammaArgmax = gammaArgmax[t] === k;
      const g = gamma[t]![k]!;
      const color = onPath
        ? tokens.color.accent
        : isGammaArgmax
          ? tokens.series[1]
          : tokens.color.borderStrong;
      return {
        id: n.id,
        x: n.x,
        y: n.y,
        label: String(k + 1),
        observed: onPath,
        color,
        radius: 9 + 12 * g,
        dimmed: !onPath && backtracking && viterbi.path[t] !== k && !isGammaArgmax,
      };
    });

  const edges = layout
    .filter((n) => n.layer > 0 && n.layer <= sweepColumn)
    .map((n) => {
      const t = n.layer;
      const k = n.index;
      const predecessor = viterbi.psi[t]![k]!;
      const onPath = backtracking && viterbi.path[t] === k && viterbi.path[t - 1] === predecessor;
      return {
        from: nodeAt(t - 1, predecessor).id,
        to: n.id,
        directed: true,
        color: onPath ? tokens.color.accent : tokens.color.inkFaint,
        width: onPath ? 2.5 : 1,
        dimmed: backtracking && !onPath,
      };
    });

  const stepCount = N + 1;
  const labels = [...Array.from({ length: N }, (_, t) => `Column ${t + 1}`), 'Backtrack the best path'];

  return (
    <div>
      <Plot width={720} height={280} xDomain={[0, 1]} yDomain={[0, 1]} label="HMM trellis: states down, time across">
        <NetworkDiagram nodes={nodes} edges={edges} directed nodeRadius={12} />
      </Plot>
      <StepThrough step={step} stepCount={stepCount} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        {backtracking
          ? 'Accent path: the single most probable sequence. Second colour: the state that is individually most probable in its own column. Where they differ, the argmax-per-column sequence is not a real contender at all.'
          : `Column ${sweepColumn + 1} of ${N}: every node's size is its posterior marginal gamma; the edge into it is the surviving (highest-scoring) predecessor.`}
      </p>
    </div>
  );
}
