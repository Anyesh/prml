import { gaussianBasis, kernelFromFeatureMap, linspace } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const CENTRES = linspace(-0.8, 0.8, 5);
const SCALE = 0.3;
const PHI = gaussianBasis(CENTRES, SCALE, { bias: false });
const KERNEL = kernelFromFeatureMap((v: readonly number[]) => PHI(v[0]!));
const GRID = linspace(-1, 1, 161);

export default function BasisToKernel() {
  const tokens = useResolvedTokens();
  const basisCurves = CENTRES.map((_, j) => GRID.map((x) => [x, PHI(x)[j]!] as const));
  const kernelCurve = GRID.map((x) => [x, KERNEL([x], [0])] as const);
  const peak = Math.max(...kernelCurve.map(([, y]) => Math.abs(y)));

  return (
    <div className="widget-grid">
      <Plot height={180} xDomain={[-1, 1]} yDomain={[0, 1.05]} label="Five Gaussian basis functions">
        <Axes x={{ label: 'x' }} y={{ label: 'phi_i(x)' }} grid />
        {basisCurves.map((points, i) => (
          <Curve key={i} points={points} color={tokens.series[i % tokens.series.length]!} width={1.5} />
        ))}
      </Plot>
      <Plot height={180} xDomain={[-1, 1]} yDomain={[-peak * 0.1, peak * 1.1]} label="The kernel these five basis functions induce, k(x, 0)">
        <Axes x={{ label: 'x' }} y={{ label: 'k(x, 0)' }} grid />
        <Curve points={kernelCurve} color={tokens.color.accent} width={2} />
      </Plot>
      <p className="widget-readout">
        {`Sum the five products phi_i(x) phi_i(0) at every x and the individual bumps above disappear into one smooth kernel below, exactly (6.10): the sum only ever needs the dot product of the two feature vectors, not the vectors read off individually.`}
      </p>
    </div>
  );
}
