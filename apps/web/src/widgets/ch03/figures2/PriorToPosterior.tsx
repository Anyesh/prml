import { useMemo } from 'react';
import {
  evalGrid,
  isotropicPrior,
  linspace,
  mvnLogPdf,
  pcg32,
  polynomialBasis,
  standardNormal,
  updatePosterior,
  type Grid2D,
  type WeightPosterior,
} from '@prml/math';
import {
  Axes,
  ContourField,
  Plot,
  ScatterField,
  computeLevels,
  flattenField,
  quantize,
  sequentialScale,
  useResolvedTokens,
} from '@prml/viz';
import '../../widgets.css';

const TRUE_W = [-0.3, 0.5] as const;
const ALPHA = 2;
const BETA = 25;
const NOISE_STD = 0.2;
const SEED = 20260317;
const OBS_COUNT = 5;
const LEVEL_COUNT = 6;
const PHI = polynomialBasis(1);
const GRID = linspace(-1, 1, 60);
const DOMAIN: readonly [number, number] = [-1, 1];

interface Observation {
  readonly x: number;
  readonly t: number;
}

function generateObservations(): Observation[] {
  const rng = pcg32(SEED);
  return Array.from({ length: OBS_COUNT }, () => {
    const x = rng.next() * 2 - 1;
    return { x, t: TRUE_W[0] + TRUE_W[1] * x + NOISE_STD * standardNormal(rng) };
  });
}

const OBSERVATIONS = generateObservations();

function densityOf(posterior: WeightPosterior): Grid2D {
  return evalGrid(GRID, GRID, (w0, w1) => Math.exp(mvnLogPdf([w0, w1], posterior)));
}

/**
 * Colours a field by the rank of its own quantile levels rather than by absolute density,
 * so the prior and posterior read as the same drawing at two different scales: contraction
 * stays the only visible difference, instead of the prior also going flat because its peak
 * sits nowhere near the posterior's.
 */
function rankFill(density: Grid2D, stops: readonly string[]): { levels: number[]; fill: (level: number) => string } {
  const flattened = flattenField(density);
  const levels = computeLevels(flattened.values, undefined, LEVEL_COUNT);
  const colors = quantize(sequentialScale([0, 1], stops), Math.max(levels.length, 1));
  const fill = (level: number) => colors[levels.indexOf(level)] ?? colors[colors.length - 1]!;
  return { levels, fill };
}

interface WeightSpacePanelProps {
  readonly density: Grid2D;
  readonly field: { levels: number[]; fill: (level: number) => string };
  readonly label: string;
  readonly caption: string;
}

function WeightSpacePanel({ density, field, label, caption }: WeightSpacePanelProps) {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label={label}>
        <ContourField data={density} levels={field.levels} fill={field.fill} color={tokens.color.inkFaint} z={-1} />
        <Axes x={{ label: 'w₀' }} y={{ label: 'w₁' }} grid />
        <ScatterField
          points={[{ x: TRUE_W[0], y: TRUE_W[1], shape: 'cross', color: tokens.color.danger, size: 7 }]}
          label={() => 'True weights'}
        />
      </Plot>
      <p className="widget-readout">{caption}</p>
    </div>
  );
}

export default function PriorToPosterior() {
  const tokens = useResolvedTokens();

  const prior = useMemo(() => isotropicPrior(2, ALPHA), []);
  const posterior = useMemo(
    () => OBSERVATIONS.reduce((p, o) => updatePosterior(p, PHI(o.x), o.t, BETA), prior),
    [prior],
  );

  const priorDensity = useMemo(() => densityOf(prior), [prior]);
  const posteriorDensity = useMemo(() => densityOf(posterior), [posterior]);
  const priorField = useMemo(() => rankFill(priorDensity, tokens.sequential), [priorDensity, tokens.sequential]);
  const posteriorField = useMemo(
    () => rankFill(posteriorDensity, tokens.sequential),
    [posteriorDensity, tokens.sequential],
  );

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16 }}>
      <WeightSpacePanel
        density={priorDensity}
        field={priorField}
        label="Prior over the two weights"
        caption="Prior, before any data."
      />
      <WeightSpacePanel
        density={posteriorDensity}
        field={posteriorField}
        label={`Posterior after ${OBS_COUNT} observations`}
        caption={`Posterior after ${OBS_COUNT} observations.`}
      />
    </div>
  );
}
