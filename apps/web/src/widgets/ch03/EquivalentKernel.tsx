import { useMemo, useState } from 'react';
import {
  designMatrix,
  equivalentKernel,
  evalGrid,
  gaussianBasis,
  linspace,
  pcg32,
  standardNormal,
  weightPosterior,
} from '@prml/math';
import { Axes, Curve, divergingScale, Heatmap, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const DATASET_SEED = 20260916;
const DATASET_SIZE = 25;
const NOISE_STD = 0.2;

interface Observation {
  readonly x: number;
  readonly t: number;
}

/** `sin(2πx)` plus Gaussian noise, `x` uniform on `[0, 1]`, matching the other ch03 widgets. */
function generateDataset(): Observation[] {
  const rng = pcg32(DATASET_SEED);
  const xs = Array.from({ length: DATASET_SIZE }, () => rng.next());
  return xs.map((x) => ({ x, t: Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng) }));
}

const DATASET = generateDataset();
const CENTRES = linspace(0, 1, 9);
const PHI = gaussianBasis(CENTRES, 0.1);
const GRID = linspace(0, 1, 120);

// Fixed rather than sliders: the point of this widget is the kernel's shape as x moves,
// not how it responds to the hyperparameters that PredictiveDistribution already covers.
const ALPHA = 2;
const BETA = 25;

const DESIGN = designMatrix(DATASET.map((p) => p.x), PHI);
const TARGETS = DATASET.map((p) => p.t);
const POSTERIOR = weightPosterior(DESIGN, TARGETS, { alpha: ALPHA, beta: BETA });
const KERNEL = equivalentKernel(POSTERIOR, BETA);

const KERNEL_FIELD = evalGrid(GRID, GRID, (x, xp) => KERNEL(PHI(x), PHI(xp)));

const KERNEL_PEAK = KERNEL_FIELD.values.reduce(
  (peak, row) => row.reduce((rowPeak, v) => Math.max(rowPeak, Math.abs(v)), peak),
  0,
);
const KERNEL_DOMAIN: readonly [number, number] = [-KERNEL_PEAK * 1.05, KERNEL_PEAK * 1.05];

function clampToUnit(x: number): number {
  return Math.min(1, Math.max(0, x));
}

export default function EquivalentKernel() {
  const [queryX, setQueryX] = useState(0.5);
  const tokens = useResolvedTokens();

  const kernelCurve = useMemo(() => {
    const phiQuery = PHI(queryX);
    return GRID.map((xp) => [xp, KERNEL(phiQuery, PHI(xp))] as const);
  }, [queryX]);

  const markerY = KERNEL(PHI(queryX), PHI(queryX));

  const weights = useMemo(() => {
    const phiQuery = PHI(queryX);
    return DATASET.map((p) => KERNEL(phiQuery, PHI(p.x)));
  }, [queryX]);
  const weightSum = weights.reduce((s, w) => s + w, 0);
  const negativeCount = weights.filter((w) => w < 0).length;

  const fill = divergingScale(KERNEL_DOMAIN, 0);

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[0, 1]} yDomain={KERNEL_DOMAIN} label="Equivalent kernel k(x, x') as a function of x'">
        <Axes x={{ label: "x'" }} y={{ label: 'k(x, x′)' }} grid zeroLine />
        <Curve points={kernelCurve} color={tokens.color.accent} width={2} />
        <ScatterField
          points={[{ x: queryX, y: markerY, id: 'query' }]}
          color={tokens.color.danger}
          size={6}
          onMove={(_, x) => setQueryX(clampToUnit(x))}
          label={() => 'Query point x, draggable'}
        />
      </Plot>

      <Plot height={300} xDomain={[0, 1]} yDomain={[0, 1]} equalAspect label="Equivalent kernel over the (x, x') square">
        <Heatmap data={KERNEL_FIELD} interpolator={fill} />
        <Axes x={{ label: 'x' }} y={{ label: "x'" }} grid />
        <Curve points={[[queryX, 0], [queryX, 1]] as const} color={tokens.color.danger} dash="dashed" width={1.5} />
        <Curve points={[[0, queryX], [1, queryX]] as const} color={tokens.color.danger} dash="dashed" width={1.5} />
      </Plot>

      <Panel columns={2} dense>
        <Slider
          label="Query point x"
          value={queryX}
          onChange={setQueryX}
          min={0}
          max={1}
          step={0.01}
          hint="Where the local smoothing weight is centred. The marker on the left is draggable too."
        />
        <p className="widget-readout">
          {`At x=${queryX.toFixed(2)}, the ${DATASET_SIZE} smoothing weights k(x, xₙ) sum to ${weightSum.toFixed(3)} ` +
            `(the book's guarantee is exactly one), and ${negativeCount} of them ${negativeCount === 1 ? 'is' : 'are'} negative: ` +
            `the equivalent kernel reweights and cancels rather than only averaging nearby points.`}
        </p>
      </Panel>
    </div>
  );
}
