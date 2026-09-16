import { markovChainLogLikelihood } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';

// Five days, rain-rain-rain-sun-sun (0 = rain, 1 = sun): a streak, which is exactly what
// an i.i.d. model cannot represent any better than any other arrangement of 3 rainy days.
const STATES = [0, 0, 0, 1, 1];
const IID_P_RAIN = 3 / 5;
const IID_PI = [IID_P_RAIN, 1 - IID_P_RAIN];
const IID_A = [
  [IID_P_RAIN, 1 - IID_P_RAIN],
  [IID_P_RAIN, 1 - IID_P_RAIN],
];
const MARKOV_PI = [0.5, 0.5];
const MARKOV_A = [
  [0.7, 0.3],
  [0.3, 0.7],
];

export default function SequenceProbabilityComparison() {
  const tokens = useResolvedTokens();
  const iidProb = Math.exp(markovChainLogLikelihood(IID_PI, IID_A, STATES));
  const markovProb = Math.exp(markovChainLogLikelihood(MARKOV_PI, MARKOV_A, STATES));

  const bars = [
    { at: 0, value: iidProb, color: tokens.color.inkMuted },
    { at: 1, value: markovProb, color: tokens.color.accent },
  ];

  return (
    <Plot width={220} height={180} xDomain={[-0.6, 1.6]} yDomain={[0, markovProb * 1.2]} label="Probability of the exact 5-day sequence under each model">
      <Axes x={{ label: 'i.i.d. / Markov' }} y={{ label: 'p(sequence)' }} grid />
      <Bars bars={bars} thickness={0.6} />
    </Plot>
  );
}
