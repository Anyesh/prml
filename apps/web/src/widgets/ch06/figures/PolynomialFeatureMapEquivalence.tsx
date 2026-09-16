import { dot } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const X = [1.5, -0.8];
const Z = [0.6, 1.2];

function phi(v: readonly number[]): number[] {
  return [v[0]! * v[0]!, Math.sqrt(2) * v[0]! * v[1]!, v[1]! * v[1]!];
}

const KERNEL_VALUE = dot(X, Z) ** 2;
const PHI_X = phi(X);
const PHI_Z = phi(Z);
const FEATURE_PRODUCT = dot(PHI_X, PHI_Z);

export default function PolynomialFeatureMapEquivalence() {
  const tokens = useResolvedTokens();
  const maxAbs = Math.max(...PHI_X.map(Math.abs), ...PHI_Z.map(Math.abs));

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={[-0.6, 5.6]} yDomain={[-maxAbs * 1.2, maxAbs * 1.2]} label="The three feature-map components for x and for z">
        <Axes x={{ label: 'feature index', ticks: [0, 1, 2, 3, 4, 5] }} y={{ label: 'phi component value' }} grid zeroLine />
        <Bars
          bars={[
            ...PHI_X.map((v, i) => ({ at: i * 2, value: v, color: tokens.series[0]! })),
            ...PHI_Z.map((v, i) => ({ at: i * 2 + 1, value: v, color: tokens.series[1]! })),
          ]}
          thickness={0.7}
        />
      </Plot>
      <p className="widget-readout">
        {`x = (${X[0]}, ${X[1]}), z = (${Z[0]}, ${Z[1]}). Direct kernel: (x^Tz)^2 = ${KERNEL_VALUE.toFixed(4)}. ` +
          `Feature map phi(x) = (x1^2, sqrt(2) x1 x2, x2^2), then phi(x)^T phi(z) = ${FEATURE_PRODUCT.toFixed(4)}: the same number, exactly (6.12), computed two different ways.`}
      </p>
    </div>
  );
}
