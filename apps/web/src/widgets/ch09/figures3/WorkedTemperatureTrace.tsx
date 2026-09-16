import { gmmResponsibilities, kmeansFit, kmeansInit, pcg32, type GmmParams } from '@prml/math';
import { Annotation, Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { faithfulLikeData } from '../data.js';
import '../../widgets.css';

const DATA = faithfulLikeData();
const K = 2;
const SEED = 20260915;
const POINT_INDEX = 25;
const LOG_EPSILON_MIN = -3;
const LOG_EPSILON_MAX = 0.7;
const SAMPLES = 50;

const MEANS = (() => {
  const rng = pcg32(SEED);
  const initial = kmeansInit(rng, DATA, K);
  return kmeansFit(DATA, initial, 10).means;
})();

function gamma1(epsilon: number): number {
  const cov = [
    [epsilon, 0],
    [0, epsilon],
  ];
  const params: GmmParams = { components: MEANS.map((mean) => ({ weight: 1 / K, mean, cov })) };
  return gmmResponsibilities(DATA[POINT_INDEX]!, params)[0]!;
}

const CURVE = Array.from({ length: SAMPLES }, (_, i) => {
  const le = LOG_EPSILON_MIN + ((LOG_EPSILON_MAX - LOG_EPSILON_MIN) * i) / (SAMPLES - 1);
  return [le, gamma1(10 ** le)] as const;
});
const MARK_EPSILONS = [1.2, 0.05];
const MARKS = MARK_EPSILONS.map((e) => [Math.log10(e), gamma1(e)] as const);

export default function WorkedTemperatureTrace() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[LOG_EPSILON_MIN, LOG_EPSILON_MAX]} yDomain={[0, 1]} label="One point's responsibility to component 1, tracked continuously as epsilon shrinks">
      <Axes x={{ label: 'log10(epsilon)' }} y={{ label: 'gamma_1' }} grid />
      <Curve points={CURVE} color={tokens.color.accent} width={2} />
      {MARKS.map(([x, y], i) => (
        <Annotation key={i} x={x} y={y} text={y.toFixed(3)} dy={-14} />
      ))}
    </Plot>
  );
}
