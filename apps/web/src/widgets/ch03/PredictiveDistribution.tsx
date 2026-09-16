import { useMemo, useState } from 'react';
import {
  designMatrix,
  dot,
  gaussianBasis,
  linspace,
  pcg32,
  predictive,
  sampleWeights,
  standardNormal,
  weightPosterior,
} from '@prml/math';
import { Axes, Band, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const DATASET_SEED = 20260916;
const DATASET_SIZE = 25;
const NOISE_STD = 0.2;

interface Observation {
  readonly x: number;
  readonly t: number;
}

/**
 * `sin(2πx)` plus Gaussian noise, `x` drawn uniformly on `[0, 1]` in the order the rng
 * produces it rather than sorted, so revealing the first few points leaves a visible gap
 * for the "outside the data" half of the readout to point at.
 */
function generateDataset(): Observation[] {
  const rng = pcg32(DATASET_SEED);
  const xs = Array.from({ length: DATASET_SIZE }, () => rng.next());
  return xs.map((x) => ({ x, t: Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng) }));
}

const DATASET = generateDataset();
const CENTRES = linspace(0, 1, 9);
const PHI = gaussianBasis(CENTRES, 0.1);
const GRID = linspace(0, 1, 150);
const SAMPLE_COUNT = 5;

/** The grid point farthest from every revealed x, where the predictive band is widest. */
function farthestFromData(xs: readonly number[]): number {
  let best = GRID[0]!;
  let bestDist = -Infinity;
  for (const candidate of GRID) {
    let minDist = Infinity;
    for (const x of xs) minDist = Math.min(minDist, Math.abs(candidate - x));
    if (minDist > bestDist) {
      bestDist = minDist;
      best = candidate;
    }
  }
  return best;
}

export default function PredictiveDistribution() {
  const [revealed, setRevealed] = useState(4);
  const [alpha, setAlpha] = useState(2);
  const [beta, setBeta] = useState(25);
  const tokens = useResolvedTokens();

  const points = useMemo(() => DATASET.slice(0, revealed), [revealed]);

  const posterior = useMemo(() => {
    const design = designMatrix(points.map((p) => p.x), PHI);
    const targets = points.map((p) => p.t);
    return weightPosterior(design, targets, { alpha, beta });
  }, [points, alpha, beta]);

  const curve = useMemo(
    () =>
      GRID.map((x) => {
        const { mean, variance } = predictive(PHI(x), posterior, beta);
        return { x, mean, std: Math.sqrt(variance) };
      }),
    [posterior, beta],
  );

  const meanPoints = curve.map((c) => [c.x, c.mean] as const);
  const bandPoints = curve.map((c) => [c.x, c.mean - c.std, c.mean + c.std] as const);

  // Reseeded from the revealed count so the drawn functions change as evidence arrives
  // but stay put while alpha or beta is dragged, which would otherwise read as noise.
  const sampleCurves = useMemo(() => {
    const draws = sampleWeights(pcg32(DATASET_SEED, revealed + 1), posterior, SAMPLE_COUNT);
    return draws.map((w) => GRID.map((x) => [x, dot(PHI(x), w)] as const));
  }, [posterior, revealed]);

  const insideX = points[Math.floor(points.length / 2)]!.x;
  const outsideX = farthestFromData(points.map((p) => p.x));
  const insideStd = Math.sqrt(predictive(PHI(insideX), posterior, beta).variance);
  const outsideStd = Math.sqrt(predictive(PHI(outsideX), posterior, beta).variance);
  const noiseFloor = 1 / Math.sqrt(beta);

  return (
    <div className="widget-grid">
      <Plot height={340} xDomain={[0, 1]} yDomain={[-2.5, 2.5]} label="Predictive distribution over the target">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Band points={bandPoints} color={tokens.color.accent} opacity={0.18} />
        {sampleCurves.map((pts, i) => (
          <Curve key={i} points={pts} color={tokens.series[1]!} width={1} opacity={0.5} />
        ))}
        <Curve points={meanPoints} color={tokens.color.accent} width={2} />
        <ScatterField
          points={points.map((p, i) => ({ x: p.x, y: p.t, id: i }))}
          color={tokens.color.ink}
          size={4}
          label={(_, i) => `Observation ${i + 1}`}
        />
        <Legend
          entries={[
            { label: 'predictive mean', color: tokens.color.accent, mark: 'line' },
            { label: '±1σ band', color: tokens.color.accent, mark: 'swatch' },
            { label: 'sampled functions', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Panel columns={2} dense>
        <Slider
          label="Revealed points"
          value={revealed}
          onChange={(v) => setRevealed(Math.round(v))}
          min={1}
          max={DATASET_SIZE}
          step={1}
          hint="How many of the 25 training points the model has seen so far."
        />
        <Slider
          label="Prior precision α"
          value={alpha}
          onChange={setAlpha}
          min={0.01}
          max={100}
          scale="log"
          hint="How tightly the prior pins the weights to zero before any data arrives."
        />
        <Slider
          label="Noise precision β"
          value={beta}
          onChange={setBeta}
          min={1}
          max={200}
          scale="log"
          hint="Assumed noise precision. The band can never narrow past 1/√β."
        />
        <p className="widget-readout">
          {`With ${points.length} of ${DATASET_SIZE} points revealed, the predictive std at x=${insideX.toFixed(2)} ` +
            `(inside the data) is ${insideStd.toFixed(3)}, closing in on the noise floor 1/√β = ${noiseFloor.toFixed(3)} ` +
            `rather than zero. At x=${outsideX.toFixed(2)}, away from any observation, it widens to ${outsideStd.toFixed(3)}.`}
        </p>
      </Panel>
    </div>
  );
}
