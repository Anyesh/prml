import { kmeansDistortion, kmeansFit, kmeansInit, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import { kmeansDemoData } from '../data.js';
import '../../widgets.css';

const DATA = kmeansDemoData();
const K = 3;
const ROUNDS = 7;
const SEED = 20260913;

function computeSeries() {
  const rng = pcg32(SEED);
  const initialMeans = kmeansInit(rng, DATA, K);
  const result = kmeansFit(DATA, initialMeans, ROUNDS);
  const afterE = result.distortionHistory;
  const afterM = result.meansHistory
    .slice(1)
    .map((means, i) => kmeansDistortion(DATA, means, result.assignmentsHistory[i]!));
  return { afterE, afterM };
}

const { afterE, afterM } = computeSeries();

export default function DistortionEM() {
  const tokens = useResolvedTokens();
  const pointsE = afterE.map((v, i) => [i + 1, v] as const);
  const pointsM = afterM.map((v, i) => [i + 1, v] as const);
  const maxY = Math.max(...afterE, ...afterM);

  return (
    <Plot height={220} xDomain={[0.5, ROUNDS + 0.5]} yDomain={[0, maxY * 1.1]} label="Distortion measured right after each E-step and right after each following M-step">
      <Axes x={{ label: 'round', ticks: [1, 2, 3, 4, 5] }} y={{ label: 'J' }} grid />
      <Curve points={pointsE} color={tokens.series[0]!} width={2} />
      <Curve points={pointsM} color={tokens.series[1]!} width={2} dash="dashed" />
      <Legend
        entries={[
          { label: 'after E-step (reassign)', color: tokens.series[0]!, mark: 'dot' },
          { label: 'after M-step (recentre)', color: tokens.series[1]!, mark: 'dashed-line' },
        ]}
      />
    </Plot>
  );
}
