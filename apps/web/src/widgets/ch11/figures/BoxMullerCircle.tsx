import { boxMullerTrace, pcg32 } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 1;

export default function BoxMullerCircle() {
  const tokens = useResolvedTokens();
  const trace = boxMullerTrace(pcg32(SEED, 1));
  const circlePoints = Array.from({ length: 129 }, (_, i) => {
    const t = (2 * Math.PI * i) / 128;
    return [Math.cos(t), Math.sin(t)] as const;
  });

  return (
    <div className="widget-grid">
      <Plot width={280} height={280} xDomain={[-1.2, 1.2]} yDomain={[-1.2, 1.2]} equalAspect label="Box-Muller attempts inside and outside the unit circle">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <ScatterField
          points={trace.attempts.map((a) => ({
            x: a.z1,
            y: a.z2,
            color: a.accepted ? tokens.color.success : tokens.color.danger,
            shape: a.accepted ? 'circle' : 'cross',
            size: 5,
          }))}
        />
        <ScatterField points={circlePoints.map(([x, y]) => ({ x, y, color: tokens.color.inkFaint, size: 1 }))} />
      </Plot>
      <p className="widget-readout">
        This fixed seed took {trace.attempts.length} attempt{trace.attempts.length === 1 ? '' : 's'} before landing
        inside the disk: r² = {trace.attempts[trace.attempts.length - 1]!.r2.toFixed(3)}. The accepted pair maps to
        y1 = {trace.y1.toFixed(3)}, y2 = {trace.y2.toFixed(3)} (11.10-11.12).
      </p>
    </div>
  );
}
