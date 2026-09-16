import { vonMisesFit } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DEG_TO_RAD = Math.PI / 180;
const ANGLES_DEG = [1, 359];
const ANGLES_RAD = ANGLES_DEG.map((d) => d * DEG_TO_RAD);
const NAIVE_MEAN_DEG = (ANGLES_DEG[0]! + ANGLES_DEG[1]!) / 2;
const FIT = vonMisesFit(ANGLES_RAD);
const CIRCULAR_MEAN_DEG = ((FIT.mu / DEG_TO_RAD) % 360 + 360) % 360;

const CIRCLE = Array.from({ length: 100 }, (_, i) => {
  const t = (2 * Math.PI * i) / 99;
  return [Math.cos(t), Math.sin(t)] as const;
});

export default function VonMisesNaiveMeanFails() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[-1.4, 1.4]} yDomain={[-1.4, 1.4]} equalAspect label="Two angles near due east, with the naive and circular means">
        <Axes x={false} y={false} />
        <Curve points={CIRCLE} color={tokens.color.inkFaint} dash="dashed" width={1} />
        <ScatterField
          points={ANGLES_RAD.map((t) => ({ x: Math.cos(t), y: Math.sin(t), color: tokens.color.ink, size: 4 }))}
        />
        <Curve points={[[0, 0], [Math.cos(NAIVE_MEAN_DEG * DEG_TO_RAD), Math.sin(NAIVE_MEAN_DEG * DEG_TO_RAD)]]} color={tokens.color.danger} width={2} />
        <Curve points={[[0, 0], [Math.cos(FIT.mu), Math.sin(FIT.mu)]]} color={tokens.color.accent} width={2} />
      </Plot>
      <p className="widget-readout">
        {`Angles 1° and 359° average to ${NAIVE_MEAN_DEG}° arithmetically (red), due west and wrong. `}
        {`vonMisesFit gives a mean direction of ${CIRCULAR_MEAN_DEG.toFixed(1)}° (blue) with κ=${FIT.kappa.toFixed(0)}, `}
        {'the direction the two points actually sit near.'}
      </p>
    </div>
  );
}
