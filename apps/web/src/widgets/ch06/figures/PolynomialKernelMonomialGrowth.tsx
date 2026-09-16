import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

function choose(n: number, k: number): number {
  let result = 1;
  for (let i = 0; i < k; i++) result = (result * (n - i)) / (i + 1);
  return result;
}

// Monomials of degree exactly M over D inputs, C(D + M - 1, M): the dimension of the
// feature space (x^T x')^M expands into, before any kernel trick avoids building it.
function monomialCount(dimension: number, degree: number): number {
  return choose(dimension + degree - 1, degree);
}

const DIMENSIONS = Array.from({ length: 20 }, (_, i) => i + 1);
const DEGREES = [2, 3, 5];

export default function PolynomialKernelMonomialGrowth() {
  const tokens = useResolvedTokens();
  const curves = DEGREES.map((degree) => DIMENSIONS.map((d) => [d, Math.log10(monomialCount(d, degree))] as const));
  const at20 = DEGREES.map((degree) => monomialCount(20, degree));

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[1, 20]} yDomain={[0, 10]} label="Feature-space dimension of a degree-M polynomial kernel against input dimension D">
        <Axes x={{ label: 'D, input dimension' }} y={{ label: 'monomials of exact degree M', format: (v) => `1e${Math.round(v)}` }} grid />
        {curves.map((points, i) => (
          <Curve key={i} points={points} color={tokens.series[i]!} width={2} />
        ))}
        <Legend entries={DEGREES.map((degree, i) => ({ label: `M = ${degree}`, color: tokens.series[i]!, mark: 'line' as const }))} placement="top-left" />
      </Plot>
      <p className="widget-readout">
        {`At D = 20 inputs, a degree-${DEGREES[0]} kernel's feature space has ${at20[0]!.toLocaleString()} dimensions, degree-${DEGREES[1]} has ${at20[1]!.toLocaleString()}, degree-${DEGREES[2]} has ${at20[2]!.toLocaleString()}. Section 3.6's explosion is still there; the kernel trick just means k(x, x') = (x^T x' + c)^M costs one dot product regardless.`}
      </p>
    </div>
  );
}
