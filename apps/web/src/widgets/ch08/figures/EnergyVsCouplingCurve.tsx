import { icmDenoise, linspace, pcg32, type Grid } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIZE = 10;
const SWEEPS = 5;

function buildNoisyPattern(): Grid {
  const rng = pcg32(20260915);
  const rows: number[][] = [];
  for (let i = 0; i < SIZE; i++) {
    const row: number[] = [];
    for (let j = 0; j < SIZE; j++) {
      const clean = Math.abs(i - j) <= 1 || Math.abs(i + j - (SIZE - 1)) <= 1 ? 1 : -1;
      row.push(rng.next() < 0.1 ? -clean : clean);
    }
    rows.push(row);
  }
  return rows;
}

const NOISY = buildNoisyPattern();
const BETAS = linspace(0.1, 3, 16);

export default function EnergyVsCouplingCurve() {
  const tokens = useResolvedTokens();
  const points: [number, number][] = BETAS.map((beta) => {
    const result = icmDenoise(NOISY, NOISY, { h: 0, beta, eta: 2.1 }, SWEEPS);
    return [beta, result.energyHistory[result.energyHistory.length - 1]!];
  });
  const values = points.map((p) => p[1]);

  return (
    <Plot
      width={320}
      height={220}
      xDomain={[0, 3.1]}
      yDomain={[Math.min(...values) * 1.1, Math.max(...values) * 1.1]}
      label="Energy after five ICM sweeps, as coupling strength increases"
    >
      <Axes x={{ label: 'beta' }} y={{ label: 'energy after 5 sweeps' }} grid />
      <Curve points={points} color={tokens.color.accent} width={2} />
    </Plot>
  );
}
