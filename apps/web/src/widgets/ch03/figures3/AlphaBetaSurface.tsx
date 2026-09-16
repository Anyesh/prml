import {
  designMatrix,
  evalGrid,
  linspace,
  logEvidence,
  maximiseEvidence,
  pcg32,
  polynomialBasis,
  standardNormal,
} from '@prml/math';
import { Annotation, Axes, ContourField, Plot, ScatterField, Trajectory, sequentialScale, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DATASET_SEED = 20260916;
const DATASET_SIZE = 25;
const NOISE_STD = 0.2;
const DEGREE = 9;
const INITIAL_HYPER = { alpha: 1e-2, beta: 1 };
const ALPHA_LOG_DOMAIN: readonly [number, number] = [-4, 3];
const BETA_LOG_DOMAIN: readonly [number, number] = [-1, 4];
const ALPHA_TICKS = [-4, -3, -2, -1, 0, 1, 2, 3];
const BETA_TICKS = [-1, 0, 1, 2, 3, 4];
const GRID_RESOLUTION = 45;

interface Observation {
  readonly x: number;
  readonly t: number;
}

/** Same 25-point noisy sinusoid and degree-9 basis as the rest of chapter 3's evidence figures. */
function generateDataset(): Observation[] {
  const rng = pcg32(DATASET_SEED);
  const xs = Array.from({ length: DATASET_SIZE }, () => rng.next());
  return xs.map((x) => ({ x, t: Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng) }));
}

const DATASET = generateDataset();
const DESIGN = designMatrix(DATASET.map((p) => p.x), polynomialBasis(DEGREE));
const TARGETS = DATASET.map((p) => p.t);

const ALPHA_LOG = linspace(ALPHA_LOG_DOMAIN[0], ALPHA_LOG_DOMAIN[1], GRID_RESOLUTION);
const BETA_LOG = linspace(BETA_LOG_DOMAIN[0], BETA_LOG_DOMAIN[1], GRID_RESOLUTION);

const SURFACE = evalGrid(ALPHA_LOG, BETA_LOG, (logAlpha, logBeta) =>
  logEvidence(DESIGN, TARGETS, { alpha: 10 ** logAlpha, beta: 10 ** logBeta }),
);

const FINAL = maximiseEvidence(DESIGN, TARGETS, INITIAL_HYPER);
const FINAL_LOG: readonly [number, number] = [Math.log10(FINAL.alpha), Math.log10(FINAL.beta)];

/**
 * Retraces the fixed-point iteration's own path by calling `maximiseEvidence` again with
 * an increasing iteration cap: each call is the exact state the real recursion reaches
 * after that many steps, so the path is read off the golden-tested function itself
 * rather than a re-derivation of its update rule.
 */
const PATH: (readonly [number, number])[] = [
  [Math.log10(INITIAL_HYPER.alpha), Math.log10(INITIAL_HYPER.beta)],
  ...Array.from({ length: FINAL.iterations }, (_, i) => {
    const step = maximiseEvidence(DESIGN, TARGETS, INITIAL_HYPER, i + 1);
    return [Math.log10(step.alpha), Math.log10(step.beta)] as const;
  }),
];

let peak = -Infinity;
let floor = Infinity;
for (const row of SURFACE.values) {
  for (const v of row) {
    if (v > peak) peak = v;
    if (v < floor) floor = v;
  }
}
const FILL = sequentialScale([floor, peak]);

export default function AlphaBetaSurface() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot
        height={220}
        xDomain={ALPHA_LOG_DOMAIN}
        yDomain={BETA_LOG_DOMAIN}
        label="Log evidence over the alpha-beta plane, with the fixed-point iteration's path from its starting point to the maximum"
      >
        <ContourField data={SURFACE} levelCount={10} color={tokens.color.inkFaint} fill={FILL} z={-2} />
        <Axes
          x={{ label: 'α', ticks: ALPHA_TICKS, format: (v) => `1e${Math.round(v)}` }}
          y={{ label: 'β', ticks: BETA_TICKS, format: (v) => `1e${Math.round(v)}` }}
          grid
        />
        <Trajectory path={PATH} color={tokens.color.accent} width={1.75} markers />
        <ScatterField
          points={[{ x: FINAL_LOG[0], y: FINAL_LOG[1], shape: 'cross', color: tokens.color.danger, size: 6 }]}
          label={() => 'evidence maximum'}
        />
        <Annotation
          x={FINAL_LOG[0]}
          y={FINAL_LOG[1]}
          text={`α=${FINAL.alpha.toExponential(1)}, β=${FINAL.beta.toFixed(1)}`}
          color={tokens.color.danger}
          anchor="start"
          dx={8}
          dy={-8}
          plate
        />
      </Plot>
      <p className="widget-readout">
        {`Evidence maximisation climbs this surface, not a formula: from α=${INITIAL_HYPER.alpha}, β=${INITIAL_HYPER.beta} it reaches the peak in ${FINAL.iterations} fixed-point steps.`}
      </p>
    </div>
  );
}
