import { mvnCovarianceEllipse } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { hmmDemo } from '../data.js';

const DEMO = hmmDemo();

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number, segments = 40) {
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const points: (readonly [number, number])[] = [];
  for (let i = 0; i <= segments; i++) {
    const t = (2 * Math.PI * i) / segments;
    const ex = rx * Math.cos(t);
    const ey = ry * Math.sin(t);
    points.push([cx + ex * cosA - ey * sinA, cy + ex * sinA + ey * cosA]);
  }
  return points;
}

export default function EmissionSamplesFigure() {
  const tokens = useResolvedTokens();

  return (
    <Plot width={360} height={320} xDomain={[-2.5, 6.5]} yDomain={[-2.5, 6.5]} equalAspect label="Samples generated from the HMM, coloured by true state">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      {DEMO.components.map((c, k) => {
        const ellipse = mvnCovarianceEllipse({ mean: [...c.mean], cov: c.cov }, 0.9);
        return (
          <Curve
            key={k}
            points={ellipsePoints(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle)}
            color={tokens.series[k]!}
            width={1.5}
            dash="dashed"
          />
        );
      })}
      <Curve points={DEMO.data.map((p) => [p[0]!, p[1]!] as const)} color={tokens.color.inkFaint} width={1} />
      <ScatterField
        points={DEMO.data.map((p, i) => ({ x: p[0]!, y: p[1]!, color: tokens.series[DEMO.trueStates[i]!]!, size: 4.5 }))}
        label={(_, i) => `t=${i + 1}, state ${DEMO.trueStates[i]! + 1}`}
      />
    </Plot>
  );
}
