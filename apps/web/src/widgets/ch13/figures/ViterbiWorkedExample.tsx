import { hmmBackwardScaled, hmmForwardScaled, hmmGammaScaled, hmmViterbi } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';

const pi = [0.69, 0.31];
const A = [
  [0.94, 0.06],
  [0.42, 0.58],
];
const B = [
  [0.19, 0.81],
  [0.74, 0.26],
  [0.15, 0.85],
];

export default function ViterbiWorkedExample() {
  const tokens = useResolvedTokens();
  const { alphaHat, c } = hmmForwardScaled(pi, A, B);
  const betaHat = hmmBackwardScaled(A, B, c);
  const gamma = hmmGammaScaled(alphaHat, betaHat);
  const perColumnBest = gamma.map((row) => row.indexOf(Math.max(...row)));
  const viterbi = hmmViterbi(pi, A, B);

  const rows = [
    { y: 1, label: 'Per-column argmax(gamma)', path: perColumnBest },
    { y: 0, label: 'Viterbi path', path: viterbi.path },
  ];

  return (
    <Plot width={320} height={140} xDomain={[-0.5, 2.5]} yDomain={[-0.5, 1.5]} label="Individually best states against the single best path">
      <Axes x={{ label: 'time step' }} y={{ label: '' }} />
      {rows.map((row) => (
        <ScatterField
          key={row.label}
          points={row.path.map((state, t) => ({ x: t, y: row.y, color: tokens.series[state]!, size: 6 }))}
          label={(_, t) => `${row.label}, t=${t + 1}`}
        />
      ))}
    </Plot>
  );
}
