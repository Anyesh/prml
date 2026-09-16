import { useState } from 'react';
import { linspace, logBinomialCoefficient } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const D_MIN = 1;
const D_MAX = 10;
const D_RANGE = linspace(D_MIN, D_MAX, D_MAX - D_MIN + 1);
const POLYNOMIAL_DEGREE = 3;
const COMPUTE_LIMIT = 1e9;
const COMPUTE_LIMIT_LOG = Math.log10(COMPUTE_LIMIT);
const Y_TICKS = [0, 3, 6, 9, 12, 15];

/** `binomial(D + degree, degree)`, PRML's count of degree-`d` polynomial terms in `D` inputs. */
function polynomialTermCount(inputDimension: number): number {
  return Math.exp(logBinomialCoefficient(inputDimension + POLYNOMIAL_DEGREE, POLYNOMIAL_DEGREE));
}

export default function BasisCountExplosion() {
  const [spacing, setSpacing] = useState(0.2);
  const tokens = useResolvedTokens();

  const gaussianCurve = D_RANGE.map((d) => [d, d * Math.log10(1 / spacing)] as const);
  const polyCurve = D_RANGE.map((d) => [d, Math.log10(polynomialTermCount(d))] as const);

  const gaussianTop = gaussianCurve[gaussianCurve.length - 1]![1];
  const yMax = Math.max(COMPUTE_LIMIT_LOG, gaussianTop) + 1;

  return (
    <div className="widget-grid">
      <Plot
        height={220}
        xDomain={[D_MIN, D_MAX]}
        yDomain={[0, yMax]}
        label="Basis functions needed against input dimension D: Gaussian bumps at fixed spacing versus degree-3 polynomial terms"
      >
        <Axes
          x={{ label: 'input dimension D', ticks: D_RANGE }}
          y={{ label: 'basis functions needed', ticks: Y_TICKS, format: (v) => `1e${Math.round(v)}` }}
          grid
        />
        <Rule y={COMPUTE_LIMIT_LOG} color={tokens.color.danger} label="1e9, a plausible compute limit" />
        <Curve points={gaussianCurve} color={tokens.series[0]!} width={2} />
        <Curve points={polyCurve} color={tokens.series[1]!} width={2} />
        <Legend
          entries={[
            { label: `Gaussian bumps, spacing s = ${spacing.toFixed(2)}`, color: tokens.series[0]!, mark: 'line' },
            { label: 'polynomial terms, degree 3', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-left"
        />
      </Plot>
      <Panel dense>
        <Slider
          label="Gaussian bump spacing s"
          value={spacing}
          onChange={setSpacing}
          min={0.05}
          max={0.5}
          scale="log"
          hint="Tighter spacing raises the base of (1/s)^D, so the Gaussian curve steepens without changing the polynomial one at all."
        />
      </Panel>
    </div>
  );
}
