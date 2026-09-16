import { Annotation, Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { laplaceApproximation, linspace, sigmoid, trapz } from '@prml/math';
import '../../widgets.css';

const STEEPNESS = 20;
const OFFSET = 4;
const GRID = linspace(-2, 4, 400);

function unnormalised(z: number): number {
  return Math.exp(-(z * z) / 2) * sigmoid(STEEPNESS * z + OFFSET);
}

const RAW = GRID.map(unnormalised);
const Z = trapz(RAW, GRID);
const TRUE_DENSITY = RAW.map((v) => v / Z);
const TRUE_MEAN = trapz(
  GRID.map((z, i) => z * TRUE_DENSITY[i]!),
  GRID,
);

const LAPLACE = laplaceApproximation(
  (z) => [-z[0]! + STEEPNESS * (1 - sigmoid(STEEPNESS * z[0]! + OFFSET))],
  (z) => {
    const s = sigmoid(STEEPNESS * z[0]! + OFFSET);
    return [[-1 - STEEPNESS * STEEPNESS * s * (1 - s)]];
  },
  [0],
);

export default function ModeVsMeanMismatch() {
  const tokens = useResolvedTokens();
  const peak = Math.max(...TRUE_DENSITY);

  return (
    <Plot height={220} xDomain={[-2, 4]} yDomain={[0, peak * 1.25]} label="The mode the Laplace approximation centres on, against the true mean">
      <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
      <Curve points={GRID.map((z, i) => [z, TRUE_DENSITY[i]!] as const)} color={tokens.color.ink} width={2} />
      <Rule x={LAPLACE.mode[0]!} color={tokens.color.accent} label="mode" />
      <Rule x={TRUE_MEAN} color={tokens.color.danger} dash label="mean" />
      <Annotation
        x={(LAPLACE.mode[0]! + TRUE_MEAN) / 2}
        y={peak * 1.1}
        text={`gap ≈ ${(TRUE_MEAN - LAPLACE.mode[0]!).toFixed(2)}`}
        color={tokens.color.inkMuted}
        anchor="middle"
        plate
      />
    </Plot>
  );
}
