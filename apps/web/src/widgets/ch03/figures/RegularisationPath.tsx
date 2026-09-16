import { useState } from 'react';
import { clamp, designMatrix, linspace, pcg32, polynomialBasis, regularisedWeights, standardNormal } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SEED = 20260320;
const N_POINTS = 15;
const NOISE_STD = 0.3;
const DEGREE = 9;
const LN_LAMBDA_MIN = -8;
const LN_LAMBDA_MAX = 8;
const LN_LAMBDA_STEPS = 61;

const BASIS = polynomialBasis(DEGREE);
const LN_GRID = linspace(LN_LAMBDA_MIN, LN_LAMBDA_MAX, LN_LAMBDA_STEPS);

function buildDataset(): { xs: number[]; ts: number[] } {
  const rng = pcg32(SEED);
  const xs: number[] = [];
  const ts: number[] = [];
  for (let i = 0; i < N_POINTS; i++) {
    const x = rng.next();
    xs.push(x);
    ts.push(Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
  }
  return { xs, ts };
}

const DATASET = buildDataset();
const DESIGN = designMatrix(DATASET.xs, BASIS);

/** `paths[j]` is coefficient `j`'s value at each `LN_GRID` entry, the whole point of the figure. */
function buildPaths(): number[][] {
  const paths: number[][] = Array.from({ length: DEGREE + 1 }, () => []);
  for (const lnLambda of LN_GRID) {
    const weights = regularisedWeights(DESIGN, DATASET.ts, Math.exp(lnLambda));
    weights.forEach((w, j) => paths[j]!.push(w));
  }
  return paths;
}

const PATHS = buildPaths();
const Y_MAX = Math.max(1, ...PATHS.flat().map(Math.abs)) * 1.1;
const Y_DOMAIN: readonly [number, number] = [-Y_MAX, Y_MAX];

export default function RegularisationPath() {
  const [lnLambda, setLnLambda] = useState(0);
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot
        height={220}
        xDomain={[LN_LAMBDA_MIN, LN_LAMBDA_MAX]}
        yDomain={Y_DOMAIN}
        label="Every degree-9 coefficient against ln lambda, with a draggable marker for the current lambda"
      >
        <Axes x={{ label: 'ln λ' }} y={{ label: 'weight' }} grid zeroLine />
        {PATHS.map((path, j) => (
          <Curve key={j} points={LN_GRID.map((ln, i) => [ln, path[i]!] as const)} color={tokens.series[j % tokens.series.length]!} width={1.25} opacity={0.85} />
        ))}
        <Curve
          points={[
            [lnLambda, Y_DOMAIN[0]],
            [lnLambda, Y_DOMAIN[1]],
          ]}
          color={tokens.color.ink}
          dash="dashed"
          width={1.5}
        />
        <ScatterField
          points={[{ x: lnLambda, y: Y_DOMAIN[1], id: 'marker' }]}
          color={tokens.color.ink}
          shape="triangle"
          size={6}
          onMove={(_, x) => setLnLambda(clamp(x, LN_LAMBDA_MIN, LN_LAMBDA_MAX))}
          label={() => 'drag to change lambda'}
        />
      </Plot>
      <p className="widget-readout">
        {`λ = ${Math.exp(lnLambda).toExponential(2)}. Drag the marker toward the right and every one of the ten coefficients is pulled toward zero together.`}
      </p>
    </div>
  );
}
