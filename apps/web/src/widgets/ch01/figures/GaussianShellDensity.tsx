import { linspace, logGamma } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIGMA = 1;
const R_GRID = linspace(0.001, 8, 200);
const D_VALUES = [1, 2, 5, 20];

function logSurfaceArea(d: number): number {
  return Math.log(2) + (d / 2) * Math.log(Math.PI) - logGamma(d / 2);
}

function radialDensity(d: number, r: number): number {
  const logP = logSurfaceArea(d) + (d - 1) * Math.log(r) - (d / 2) * Math.log(2 * Math.PI * SIGMA * SIGMA) - (r * r) / (2 * SIGMA * SIGMA);
  return Math.exp(logP);
}

export default function GaussianShellDensity() {
  const tokens = useResolvedTokens();
  const curves = D_VALUES.map((d) => R_GRID.map((r) => [r, radialDensity(d, r)] as const));
  const peakAt = (d: number) => Math.sqrt(Math.max(d - 1, 0)) * SIGMA;

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[0, 8]} yDomain={[0, 1]} label="Radial probability density of a unit-variance Gaussian, by dimension">
        <Axes x={{ label: 'r' }} y={{ label: 'p(r)' }} grid />
        {curves.map((pts, i) => (
          <Curve key={D_VALUES[i]} points={pts} color={tokens.series[i % tokens.series.length]!} width={2} />
        ))}
        <Legend entries={D_VALUES.map((d, i) => ({ label: `D = ${d}, peak at r = ${peakAt(d).toFixed(2)}`, color: tokens.series[i % tokens.series.length]!, mark: 'line' }))} placement="top-right" />
      </Plot>
      <p className="widget-readout">
        {`At D=1 the density peaks at the origin. By D=20 it peaks near r=${peakAt(20).toFixed(2)}, matching the sqrt(D) sigma rule of thumb: almost none of a high-dimensional Gaussian's mass sits near its own mean.`}
      </p>
    </div>
  );
}
