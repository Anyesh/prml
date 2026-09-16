import { designMatrix, gaussianBasis, linspace, maximiseEvidence, pcg32, standardNormal } from '@prml/math';
import { Annotation, Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DATASET_SEED = 20260916;
const N = 15;
const NOISE_STD = 0.2;
const BASIS_SCALE = 0.1;
const MAX_M = 40;
const INITIAL_HYPER = { alpha: 1e-2, beta: 1 };

interface Observation {
  readonly x: number;
  readonly t: number;
}

function generateDataset(): Observation[] {
  const rng = pcg32(DATASET_SEED);
  const xs = Array.from({ length: N }, () => rng.next());
  return xs.map((x) => ({ x, t: Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng) }));
}

const DATASET = generateDataset();
const XS = DATASET.map((p) => p.x);
const TARGETS = DATASET.map((p) => p.t);

interface EffectivePoint {
  readonly m: number;
  readonly gamma: number;
}

/**
 * A few small M produce a basis close enough to singular that the fixed-point iteration
 * does not settle within its iteration budget; those M are left out of the curve rather
 * than plotted, because an unconverged gamma is not a stable estimate of anything.
 */
const CURVE: EffectivePoint[] = [];
for (let m = 1; m <= MAX_M; m++) {
  const centres = linspace(0, 1, m);
  const design = designMatrix(XS, gaussianBasis(centres, BASIS_SCALE, { bias: false }));
  const result = maximiseEvidence(design, TARGETS, INITIAL_HYPER);
  if (result.converged) CURVE.push({ m, gamma: result.effectiveParameters });
}

const CEILING = CURVE[CURVE.length - 1]!.gamma;
const CEILING_SHARE = Math.round((CEILING / N) * 100);

export default function EffectiveVersusNominal() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot
        height={220}
        xDomain={[0, MAX_M]}
        yDomain={[0, MAX_M]}
        label="Nominal parameter count M against the effective number gamma, sweeping M from 1 to well past N"
      >
        <Axes x={{ label: 'M, basis functions offered' }} y={{ label: 'parameter count' }} grid />
        <Rule y={N} color={tokens.color.inkFaint} label={`N = ${N} observations`} />
        <Curve points={[[0, 0], [MAX_M, MAX_M]] as const} color={tokens.color.inkMuted} dash="dashed" width={1.25} />
        <Curve points={CURVE.map((p) => [p.m, p.gamma] as const)} color={tokens.color.accent} width={2} />
        <Annotation
          x={MAX_M}
          y={CEILING}
          text={`γ → ${CEILING.toFixed(1)}`}
          color={tokens.color.accent}
          anchor="end"
          dy={-10}
          plate
        />
        <Legend
          entries={[
            { label: 'nominal, M', color: tokens.color.inkMuted, mark: 'dashed-line' },
            { label: 'effective, γ', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-left"
        />
      </Plot>
      <p className="widget-readout">
        {`M rises in a straight line while γ bends over early and flattens near ${CEILING.toFixed(1)}, ` +
          `about ${CEILING_SHARE}% of N = ${N}, not N itself: the hard bound from PRML 3.91 is N, but ` +
          'this basis is redundant enough that the data cannot pin down anywhere near that many directions.'}
      </p>
    </div>
  );
}
