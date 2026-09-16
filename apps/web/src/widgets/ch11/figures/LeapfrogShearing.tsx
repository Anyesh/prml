import { leapfrogStep } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIGMA2 = 1.5;
function gradEnergyFn(z: readonly number[]): number[] {
  return [z[0]! / SIGMA2];
}

function shoelaceArea(points: readonly (readonly [number, number])[]): number {
  let sum = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i]!;
    const [x2, y2] = points[(i + 1) % points.length]!;
    sum += x1 * y2 - x2 * y1;
  }
  return Math.abs(sum) / 2;
}

export default function LeapfrogShearing() {
  const tokens = useResolvedTokens();
  const corners: [number, number][] = [
    [0.4, 0.4],
    [0.6, 0.4],
    [0.6, 0.6],
    [0.4, 0.6],
  ];
  const moved = corners.map(([z, r]) => leapfrogStep({ z: [z], r: [r] }, gradEnergyFn, 0.5));
  const movedPoints = moved.map((s) => [s.z[0]!, s.r[0]!] as [number, number]);

  const close = (pts: readonly (readonly [number, number])[]) => [...pts, pts[0]!];
  const areaBefore = shoelaceArea(corners);
  const areaAfter = shoelaceArea(movedPoints);

  return (
    <div className="widget-grid">
      <Plot width={340} height={280} xDomain={[-1.2, 1.2]} yDomain={[-1.2, 1.2]} equalAspect label="A small square region of phase space, sheared by one leapfrog step">
        <Axes x={{ label: 'z' }} y={{ label: 'r' }} grid zeroLine />
        <Curve points={close(corners)} color={tokens.color.inkMuted} width={1.5} dash="dashed" />
        <Curve points={close(movedPoints)} color={tokens.series[0]!} width={2} />
        <ScatterField points={corners.map(([z, r]) => ({ x: z, y: r, color: tokens.color.inkMuted, size: 3 }))} />
        <ScatterField points={movedPoints.map(([z, r]) => ({ x: z, y: r, color: tokens.series[0]!, size: 3 }))} />
      </Plot>
      <p className="widget-readout">
        Area before: {areaBefore.toFixed(4)}. Area after one leapfrog step: {areaAfter.toFixed(4)}. The square becomes
        a parallelogram, not a square, but its area is unchanged (PRML 11.62's divergence-free flow, exactly).
      </p>
    </div>
  );
}
