import { linspace, logGamma } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const D_GRID = linspace(1, 30, 60);

function logVolumeRatio(d: number): number {
  return (d / 2) * Math.log(Math.PI) - Math.log(d) - (d - 1) * Math.log(2) - logGamma(d / 2);
}

const CURVE = D_GRID.map((d) => [d, Math.exp(logVolumeRatio(d))] as const);
const AT_5 = Math.exp(logVolumeRatio(5));
const AT_20 = Math.exp(logVolumeRatio(20));

export default function SphereInCubeCorners() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[1, 30]} yDomain={[0, 0.85]} label="Ratio of an inscribed sphere's volume to its enclosing cube's volume, against dimension">
        <Axes x={{ label: 'D' }} y={{ label: 'volume ratio' }} grid />
        <Curve points={CURVE} color={tokens.color.accent} width={2} />
      </Plot>
      <p className="widget-readout">
        {`The ratio falls from ${AT_5.toFixed(3)} at D=5 to ${AT_20.toExponential(2)} at D=20. Almost every corner of a high-dimensional cube sits outside any sphere touching its faces: volume collects in spikes at the corners, not in the middle.`}
      </p>
    </div>
  );
}
