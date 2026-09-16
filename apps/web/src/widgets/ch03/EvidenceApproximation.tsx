import { useMemo, useState } from 'react';
import {
  designMatrix,
  dot,
  maximiseEvidence,
  pcg32,
  polynomialBasis,
  standardNormal,
  weightPosterior,
} from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
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
const XS = DATASET.map((p) => p.x);
const TARGETS = DATASET.map((p) => p.t);
const GRID = Array.from({ length: 150 }, (_, i) => i / 149);
const DEGREES = Array.from({ length: 10 }, (_, i) => i);
const INITIAL_HYPER = { alpha: 1e-2, beta: 1 };

export default function EvidenceApproximation() {
  const [degree, setDegree] = useState(3);
  const tokens = useResolvedTokens();

  // Re-optimised per M rather than reusing one alpha/beta, because a fixed pair would
  // flatter low-order models or starve high-order ones and the peak would stop being honest.
  const evidenceCurve = useMemo(
    () =>
      DEGREES.map((m) => {
        const design = designMatrix(XS, polynomialBasis(m));
        return { m, ...maximiseEvidence(design, TARGETS, INITIAL_HYPER) };
      }),
    [],
  );

  const current = evidenceCurve[degree]!;
  const bestM = evidenceCurve.reduce((best, e) => (e.logEvidence > best.logEvidence ? e : best)).m;

  const currentFit = useMemo(() => {
    const design = designMatrix(XS, polynomialBasis(degree));
    return weightPosterior(design, TARGETS, { alpha: current.alpha, beta: current.beta });
  }, [degree, current]);

  const fitPoints = GRID.map((x) => [x, dot(polynomialBasis(degree)(x), currentFit.mean)] as const);

  const evidenceValues = evidenceCurve.map((e) => e.logEvidence);
  const evidenceSpan = Math.max(...evidenceValues) - Math.min(...evidenceValues) || 1;
  const evidencePadding = evidenceSpan * 0.1;
  const evidenceDomain: readonly [number, number] = [
    Math.min(...evidenceValues) - evidencePadding,
    Math.max(...evidenceValues) + evidencePadding,
  ];

  const evidenceLine = evidenceCurve.map((e) => [e.m, e.logEvidence] as const);
  const evidenceMarks = evidenceCurve.map((e) => ({
    x: e.m,
    y: e.logEvidence,
    id: e.m,
    color: e.m === degree ? tokens.color.accent : e.m === bestM ? tokens.color.danger : tokens.color.inkFaint,
    size: e.m === degree ? 6 : e.m === bestM ? 5 : 3.5,
  }));

  const nominalParameters = degree + 1;
  const determinedShare = Math.round((current.effectiveParameters / nominalParameters) * 100);

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[0, 9]} yDomain={evidenceDomain} label="Log evidence against polynomial degree M">
        <Axes x={{ label: 'M', ticks: DEGREES }} y={{ label: 'log evidence' }} grid />
        <Curve points={evidenceLine} color={tokens.color.inkMuted} width={1.25} />
        <ScatterField points={evidenceMarks} label={(p) => `M = ${p.x}`} />
        <Legend
          entries={[
            { label: 'current M', color: tokens.color.accent, mark: 'dot' },
            { label: 'evidence peak', color: tokens.color.danger, mark: 'dot' },
          ]}
          placement="bottom-right"
        />
      </Plot>

      <Plot height={300} xDomain={[0, 1]} yDomain={[-2, 2]} label="Polynomial fit at the current degree">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={fitPoints} color={tokens.color.accent} width={2} />
        <ScatterField
          points={DATASET.map((p, i) => ({ x: p.x, y: p.t, id: i }))}
          color={tokens.color.ink}
          size={4}
          label={(_, i) => `Observation ${i + 1}`}
        />
      </Plot>

      <Panel columns={2} dense>
        <Slider
          label="Polynomial degree M"
          value={degree}
          onChange={(v) => setDegree(Math.round(v))}
          min={0}
          max={9}
          step={1}
          hint="Number of basis functions minus one. The evidence, not a validation set, picks the best value."
        />
        <p className="widget-readout">
          {`At M=${degree}, evidence maximisation settles on α=${current.alpha.toExponential(2)}, β=${current.beta.toFixed(2)}, ` +
            `with γ=${current.effectiveParameters.toFixed(2)} effective parameters out of ${nominalParameters} nominal ones ` +
            `(the data itself determines about ${determinedShare}% of them). The evidence peaks at M=${bestM}.`}
        </p>
      </Panel>
    </div>
  );
}
