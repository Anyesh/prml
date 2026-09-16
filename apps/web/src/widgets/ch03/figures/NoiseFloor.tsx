import { linspace, pcg32, standardNormal } from '@prml/math';
import { Axes, Band, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260317;
const N_POINTS = 60;
const NOISE_STD = 0.35;
const GRID = linspace(0, 1, 100);

function conditionalMean(x: number): number {
  return Math.sin(2 * Math.PI * x);
}

const BAND = GRID.map((x) => [x, conditionalMean(x) - NOISE_STD, conditionalMean(x) + NOISE_STD] as const);

function sampleTargets(): { x: number; t: number }[] {
  const rng = pcg32(SEED);
  const out: { x: number; t: number }[] = [];
  for (let i = 0; i < N_POINTS; i++) {
    const x = rng.next();
    out.push({ x, t: conditionalMean(x) + NOISE_STD * standardNormal(rng) });
  }
  return out;
}

const TARGETS = sampleTargets();

export default function NoiseFloor() {
  const tokens = useResolvedTokens();

  return (
    <Plot
      height={220}
      xDomain={[0, 1]}
      yDomain={[-1.8, 1.8]}
      label="Conditional mean h(x) with target scatter and a one standard deviation noise band"
    >
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <Band points={BAND} color={tokens.color.accent} opacity={0.15} />
      <Curve points={GRID.map((x) => [x, conditionalMean(x)] as const)} color={tokens.color.accent} width={2} />
      <ScatterField
        points={TARGETS.map((p, i) => ({ x: p.x, y: p.t, id: i }))}
        color={tokens.color.inkMuted}
        size={2.5}
        opacity={0.7}
        label={() => 'observation'}
      />
      <Legend
        entries={[
          { label: 'h(x) = E[t|x]', color: tokens.color.accent, mark: 'line' },
          { label: '±1 noise std', color: tokens.color.accent, mark: 'swatch' },
        ]}
        placement="top-right"
      />
    </Plot>
  );
}
