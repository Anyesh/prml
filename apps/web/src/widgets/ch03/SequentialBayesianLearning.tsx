import { useMemo, useState } from 'react';
import {
  evalGrid,
  isotropicPrior,
  linspace,
  mvnLogPdf,
  pcg32,
  polynomialBasis,
  sampleWeights,
  updatePosterior,
  type WeightPosterior,
} from '@prml/math';
import {
  Axes,
  ClickSurface,
  ContourField,
  Curve,
  Legend,
  Plot,
  ScatterField,
  sequentialScale,
  useResolvedTokens,
} from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'Sequential Bayesian learning';
export const caption =
  'Click in the right panel to add an observation. Watch the posterior in weight space contract around the true value, and the sampled lines converge on it.';
export const figure = '3.7';

/** The generating line of PRML figure 3.7, which the weight-space panel marks as a target. */
const TRUE_W: readonly [number, number] = [-0.3, 0.5];

const PHI = polynomialBasis(1);
const GRID = linspace(-1, 1, 72);
const LINE_X = linspace(-1, 1, 2);
const SAMPLE_COUNT = 6;

interface Observation {
  readonly x: number;
  readonly t: number;
}

function posteriorAfter(points: readonly Observation[], alpha: number, beta: number): WeightPosterior {
  let posterior = isotropicPrior(2, alpha);
  for (const point of points) posterior = updatePosterior(posterior, PHI(point.x), point.t, beta);
  return posterior;
}

export default function SequentialBayesianLearning() {
  const [points, setPoints] = useState<Observation[]>([]);
  const [alpha, setAlpha] = useState(2);
  const [beta, setBeta] = useState(25);
  const tokens = useResolvedTokens();

  const posterior = useMemo(() => posteriorAfter(points, alpha, beta), [points, alpha, beta]);

  const density = useMemo(
    () => evalGrid(GRID, GRID, (w0, w1) => Math.exp(mvnLogPdf([w0, w1], posterior))),
    [posterior],
  );

  // Reseeded from the observation count so the drawn lines change as evidence arrives but
  // stay put while a slider moves, which would otherwise read as the fit being unstable.
  const lines = useMemo(
    () => sampleWeights(pcg32(20260916, points.length + 1), posterior, SAMPLE_COUNT),
    [posterior, points.length],
  );

  // The posterior's peak density grows without bound as it contracts, so the colour scale
  // must follow the current maximum rather than a fixed range, or every panel after the
  // third observation saturates to one flat colour.
  const fill = useMemo(() => {
    let peak = 0;
    for (const row of density.values) for (const v of row) if (v > peak) peak = v;
    return sequentialScale([0, peak]);
  }, [density]);

  const marks = points.map((p, i) => ({ x: p.x, y: p.t, id: i }));

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[-1, 1]} yDomain={[-1, 1]} equalAspect label="Posterior over the two weights">
        <ContourField data={density} levelCount={9} color={tokens.color.inkFaint} fill={fill} z={-2} />
        <Axes x={{ label: 'w₀' }} y={{ label: 'w₁' }} grid />
        <ScatterField
          points={[{ x: TRUE_W[0], y: TRUE_W[1], shape: 'cross', color: tokens.color.danger, size: 7 }]}
          label={() => 'True weights'}
        />
        <Legend
          entries={[{ label: 'true w', color: tokens.color.danger, mark: 'dot' }]}
          placement="top-right"
        />
      </Plot>

      <Plot height={300} xDomain={[-1, 1]} yDomain={[-1, 1]} label="Data space with lines sampled from the posterior">
        <ClickSurface
          onClick={(x, t) => setPoints((current) => [...current, { x, t }])}
          cursor="copy"
        />
        {lines.map((w, i) => (
          <Curve
            key={i}
            points={LINE_X.map((x) => [x, w[0]! + w[1]! * x] as const)}
            color={tokens.color.accent}
            opacity={0.55}
            width={1.25}
          />
        ))}
        <Curve
          points={LINE_X.map((x) => [x, TRUE_W[0] + TRUE_W[1] * x] as const)}
          color={tokens.color.danger}
          dash="dashed"
          width={1.5}
        />
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <ScatterField points={marks} color={tokens.color.ink} size={4} label={(_, i) => `Observation ${i + 1}`} />
      </Plot>

      <Panel columns={2} dense>
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
          hint="How much each observation is trusted. Lower β, slower contraction."
        />
        <p className="widget-readout">
          {points.length === 0
            ? 'No observations yet, so this is the prior: every line it draws is equally plausible.'
            : `${points.length} observation${points.length === 1 ? '' : 's'}.`}
        </p>
        <button type="button" className="widget-reset" onClick={() => setPoints([])}>
          Clear observations
        </button>
      </Panel>
    </div>
  );
}
