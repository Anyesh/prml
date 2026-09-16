import { gmmResponsibilities, kmeansAssign, kmeansFit, kmeansInit, pcg32, type GmmParams } from '@prml/math';
import { Annotation, Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { faithfulLikeData } from '../data.js';
import '../../widgets.css';

const DATA = faithfulLikeData();
const K = 2;
const SEED = 20260915;
const POINT_INDEX = 25;

const MEANS = (() => {
  const rng = pcg32(SEED);
  const initial = kmeansInit(rng, DATA, K);
  return kmeansFit(DATA, initial, 10).means;
})();

function respAt(epsilon: number): number[] {
  const cov = [
    [epsilon, 0],
    [0, epsilon],
  ];
  const params: GmmParams = { components: MEANS.map((mean) => ({ weight: 1 / K, mean, cov })) };
  return gmmResponsibilities(DATA[POINT_INDEX]!, params);
}

const SOFT = respAt(1.2);
const NEAR_HARD = respAt(0.01);
const HARD_INDEX = kmeansAssign([DATA[POINT_INDEX]!], MEANS)[0]!;

export default function SoftIndicatorBars() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[-0.5, 1.5]} yDomain={[0, 1]} label="One point's responsibility at a large and a near-zero shared variance, against its hard K-means indicator">
      <Axes x={{ label: 'component', ticks: [0, 1], format: (v) => `k=${v + 1}` }} y={{ label: 'value' }} grid />
      <ScatterField points={SOFT.map((r, k) => ({ x: k - 0.12, y: r, color: tokens.color.accent, size: 6 }))} />
      <ScatterField points={NEAR_HARD.map((r, k) => ({ x: k + 0.12, y: r, color: tokens.color.danger, size: 6 }))} />
      <ScatterField points={[{ x: HARD_INDEX, y: 1, color: tokens.color.ink, shape: 'ring', size: 10 }]} />
      <Annotation x={-0.12} y={SOFT[0]!} text={SOFT[0]!.toFixed(2)} dy={-14} color={tokens.color.accent} />
      <Annotation x={0.12} y={NEAR_HARD[0]!} text={NEAR_HARD[0]!.toFixed(2)} dy={-14} color={tokens.color.danger} />
    </Plot>
  );
}
