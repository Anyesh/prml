import { gmmResponsibilities, type GmmParams } from '@prml/math';
import { Annotation, Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const PARAMS: GmmParams = {
  components: [
    { weight: 0.6, mean: [0, 0], cov: [[1, 0.3], [0.3, 1]] },
    { weight: 0.4, mean: [3, 3], cov: [[1.5, -0.2], [-0.2, 0.5]] },
  ],
};
const POINT: readonly [number, number] = [1.5, 1.5];
const RESPONSIBILITIES = gmmResponsibilities([...POINT], PARAMS);

export default function ResponsibilityWorked() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={180} xDomain={[-0.5, 1.5]} yDomain={[0, 1]} label="Responsibility each component takes for the point (1.5, 1.5)">
      <Axes x={{ label: 'component', ticks: [0, 1], format: (v) => `k=${v + 1}` }} y={{ label: 'gamma(z_k)' }} grid />
      <ScatterField
        points={RESPONSIBILITIES.map((r, k) => ({ x: k, y: r, color: tokens.series[k]!, size: 6 + r * 22 }))}
        label={(_, k) => `Component ${k + 1}: ${RESPONSIBILITIES[k]!.toFixed(3)}`}
      />
      {RESPONSIBILITIES.map((r, k) => (
        <Annotation key={k} x={k} y={r} text={r.toFixed(3)} dy={-18} color={tokens.series[k]} />
      ))}
    </Plot>
  );
}
