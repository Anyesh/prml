import { useMemo, useState } from 'react';
import { laplaceApproximation, linspace, normalPdf, sigmoid, trapz } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const OFFSET = 4;
const GRID = linspace(-2, 4, 400);

function unnormalised(z: number, steepness: number): number {
  return Math.exp(-(z * z) / 2) * sigmoid(steepness * z + OFFSET);
}

function fitLaplace(steepness: number) {
  return laplaceApproximation(
    (z) => [-z[0]! + steepness * (1 - sigmoid(steepness * z[0]! + OFFSET))],
    (z) => {
      const s = sigmoid(steepness * z[0]! + OFFSET);
      return [[-1 - steepness * steepness * s * (1 - s)]];
    },
    [0],
  );
}

export default function LaplaceSkewedPosterior1D() {
  const [steepness, setSteepness] = useState(20);
  const tokens = useResolvedTokens();

  const laplace = useMemo(() => fitLaplace(steepness), [steepness]);

  const { trueDensity, trueMean } = useMemo(() => {
    const raw = GRID.map((z) => unnormalised(z, steepness));
    const z = trapz(raw, GRID);
    const density = raw.map((v) => v / z);
    const m = trapz(
      GRID.map((zi, i) => zi * density[i]!),
      GRID,
    );
    return { trueDensity: density, trueMean: m };
  }, [steepness]);

  const laplaceDensity = GRID.map((z) => normalPdf(z, { mu: laplace.mode[0]!, sigma2: laplace.covariance[0]![0]! }));

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[-2, 4]} yDomain={[0, Math.max(...trueDensity, ...laplaceDensity) * 1.15]} label="PRML figure 4.14's density and its Laplace approximation">
        <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
        <Curve points={GRID.map((z, i) => [z, trueDensity[i]!] as const)} color={tokens.color.ink} width={2} />
        <Curve points={GRID.map((z, i) => [z, laplaceDensity[i]!] as const)} color={tokens.color.accent} width={2} dash="dashed" />
        <Rule x={laplace.mode[0]!} color={tokens.color.accent} label="mode" />
        <Rule x={trueMean} color={tokens.color.ink} dash label="mean" />
        <Legend
          entries={[
            { label: 'p(z) ∝ exp(−z²/2) σ(kz + 4)', color: tokens.color.ink, mark: 'line' },
            { label: 'Laplace N(mode, 1/precision)', color: tokens.color.accent, mark: 'dashed-line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <Slider
        label="Steepness k"
        value={steepness}
        onChange={setSteepness}
        min={1}
        max={30}
        step={1}
        hint="k = 20 is PRML's own figure 4.14. Small k barely skews the Gaussian; large k makes the sigmoid nearly a step and the skew is unmistakable."
      />
      <p className="widget-readout">
        {`Mode ${laplace.mode[0]!.toFixed(3)}, true mean ${trueMean.toFixed(3)}, Laplace variance ${laplace.covariance[0]![0]!.toFixed(3)}. The Laplace fit centres on the mode, not the mean, and the gap between the two vertical lines is exactly what a Gaussian summary throws away.`}
      </p>
    </div>
  );
}
