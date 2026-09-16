import { kmeansFit, kmeansInit, pcg32 } from '@prml/math';
import { Annotation, Axes, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { kmeansDemoData } from '../data.js';
import '../../widgets.css';

const DATA = kmeansDemoData();
const K = 3;
const ROUNDS = 8;
const BASE_SEED = 20260914;
const SEED_COUNT = 6;

function finalDistortions(): number[] {
  const values: number[] = [];
  for (let s = 0; s < SEED_COUNT; s++) {
    const rng = pcg32(BASE_SEED, s + 1);
    const initialMeans = kmeansInit(rng, DATA, K);
    const result = kmeansFit(DATA, initialMeans, ROUNDS);
    values.push(result.distortionHistory[result.distortionHistory.length - 1]!);
  }
  return values;
}

const DISTORTIONS = finalDistortions();
const BEST = Math.min(...DISTORTIONS);

export default function LocalMinimaSeeds() {
  const tokens = useResolvedTokens();
  const maxY = Math.max(...DISTORTIONS) * 1.15;

  return (
    <Plot height={220} xDomain={[-0.5, SEED_COUNT - 0.5]} yDomain={[0, maxY]} label="Final distortion reached from six different random starts">
      <Axes x={{ label: 'random start', ticks: DISTORTIONS.map((_, i) => i) }} y={{ label: 'final J' }} grid />
      <Rule y={BEST} color={tokens.color.success} label="best found" />
      <ScatterField
        points={DISTORTIONS.map((d, i) => ({ x: i, y: d, color: d <= BEST + 1e-9 ? tokens.color.success : tokens.color.danger }))}
        size={5.5}
        label={(_, i) => `Start ${i + 1}: J = ${DISTORTIONS[i]!.toFixed(3)}`}
      />
      {DISTORTIONS.map((d, i) => (
        <Annotation key={i} x={i} y={d} text={d.toFixed(2)} dy={-14} />
      ))}
    </Plot>
  );
}
