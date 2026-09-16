import { linspace, vonMisesPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const THETA = linspace(0, 2 * Math.PI, 200);
const PARAMS = [
  { mu: Math.PI / 4, kappa: 5 },
  { mu: (3 * Math.PI) / 4, kappa: 1 },
];
const CIRCLE = THETA.map((t) => [Math.cos(t), Math.sin(t)] as const);

function petal(params: { mu: number; kappa: number }) {
  return THETA.map((t) => {
    const r = 1 + 0.4 * vonMisesPdf(t, params);
    return [r * Math.cos(t), r * Math.sin(t)] as const;
  });
}

export default function VonMisesCircle() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={240} xDomain={[-1.8, 1.8]} yDomain={[-1.8, 1.8]} equalAspect label="Von Mises density drawn as a radial bulge on the unit circle, two concentrations">
        <Axes x={false} y={false} />
        <Curve points={CIRCLE} color={tokens.color.inkFaint} dash="dashed" width={1} />
        <Curve points={petal(PARAMS[0]!)} color={tokens.series[0]!} width={2} />
        <Curve points={petal(PARAMS[1]!)} color={tokens.series[1]!} width={2} />
        <Legend
          entries={[
            { label: 'κ=5', color: tokens.series[0]!, mark: 'line' },
            { label: 'κ=1', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {'Radius above the dashed unit circle is density at that angle (eq. 2.179): higher κ pulls the bulge tighter '}
        {'around its mean direction, matching a Gaussian of variance 1/κ for large κ but staying periodic for any κ.'}
      </p>
    </div>
  );
}
