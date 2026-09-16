import { useMemo, useState } from 'react';
import { mvnSample, pcg32, standardNormal, variationalLogisticFit } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'Decision boundaries drawn from the posterior, not just the mean';
export const caption =
  'Increase the number of sampled boundaries. Where the classes overlap, sampled lines fan out; far from the data, they barely move.';
export const figure = '10.13';

const SEED = 20260917;
const N_PER_CLASS = 14;
const DOMAIN: readonly [number, number] = [-4, 4];
const PRIOR = { mean: [0, 0, 0], cov: [[4, 0, 0], [0, 4, 0], [0, 0, 4]] };
const ROUNDS = 8;

function syntheticData() {
  const rng = pcg32(SEED);
  const points: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < N_PER_CLASS; i++) {
    points.push([-1.3 + 0.7 * standardNormal(rng), -1 + 0.7 * standardNormal(rng)]);
    labels.push(0);
  }
  for (let i = 0; i < N_PER_CLASS; i++) {
    points.push([1.3 + 0.7 * standardNormal(rng), 1 + 0.7 * standardNormal(rng)]);
    labels.push(1);
  }
  return { points, labels };
}

const { points, labels } = syntheticData();
const DESIGN = points.map((p) => [1, p[0]!, p[1]!]);

function boundaryLine(w: readonly number[]): (readonly [number, number])[] {
  const [w0, w1, w2] = w as [number, number, number];
  if (Math.abs(w2) < 1e-6) return [];
  return DOMAIN.map((x1) => [x1, -(w0 + w1 * x1) / w2] as const);
}

export default function DecisionBoundaryUncertainty() {
  const [sampleCount, setSampleCount] = useState(5);
  const tokens = useResolvedTokens();

  const posterior = useMemo(() => {
    const xiInit = new Array(DESIGN.length).fill(1);
    const fit = variationalLogisticFit(DESIGN, labels, PRIOR, xiInit, ROUNDS);
    return fit.posteriorHistory[fit.posteriorHistory.length - 1]!;
  }, []);

  const sampledLines = useMemo(() => {
    const rng = pcg32(SEED + 1);
    return Array.from({ length: sampleCount }, () => boundaryLine(mvnSample(rng, posterior)));
  }, [posterior, sampleCount]);

  const meanLine = boundaryLine(posterior.mean);

  return (
    <div>
      <Plot width={320} height={320} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Sampled posterior decision boundaries over the training data">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
        {sampledLines.map((line, i) => (
          <Curve key={i} points={line} color={tokens.color.inkFaint} width={1} />
        ))}
        <Curve points={meanLine} color={tokens.color.accent} width={2.5} />
        <ScatterField
          points={points.map((p, i) => ({ x: p[0]!, y: p[1]!, color: labels[i] === 1 ? tokens.series[0]! : tokens.series[1]!, size: 4 }))}
        />
      </Plot>
      <Slider
        label="sampled boundaries"
        value={sampleCount}
        onChange={(v) => setSampleCount(Math.round(v))}
        min={1}
        max={25}
        step={1}
        format={(v) => String(Math.round(v))}
        hint="More samples make the spread of plausible boundaries, not just the mean one, visible."
      />
    </div>
  );
}
