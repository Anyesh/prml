import {
  designMatrix,
  dot,
  logDet,
  logEvidence,
  matvec,
  pcg32,
  polynomialBasis,
  standardNormal,
  vecSub,
  weightPosterior,
} from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const DATASET_SEED = 20260512;
const DATASET_SIZE = 25;
const NOISE_STD = 0.2;
const HYPER = { alpha: 5e-3, beta: 1 / (NOISE_STD * NOISE_STD) };
const DEGREES = Array.from({ length: 10 }, (_, i) => i);

interface Observation {
  readonly x: number;
  readonly t: number;
}

function generateDataset(): Observation[] {
  const rng = pcg32(DATASET_SEED);
  const xs = Array.from({ length: DATASET_SIZE }, () => rng.next());
  return xs.map((x) => ({ x, t: Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng) }));
}

const DATASET = generateDataset();
const XS = DATASET.map((p) => p.x);
const TARGETS = DATASET.map((p) => p.t);

interface EvidenceTermsAt {
  readonly m: number;
  readonly fit: number;
  readonly occam: number;
  readonly total: number;
}

/**
 * Splits PRML 3.86 into its own two competing pieces, since `logEvidence` only returns
 * their sum: the fit term `-E(m_N)` and the Occam term `-½ln|A|` on the same posterior
 * precision `A` that `weightPosterior` already produces.
 */
function evidenceTermsAt(m: number): EvidenceTermsAt {
  const design = designMatrix(XS, polynomialBasis(m));
  const { mean, precision } = weightPosterior(design, TARGETS, HYPER);
  const residual = vecSub(matvec(design, mean), TARGETS);
  const eMn = 0.5 * HYPER.beta * dot(residual, residual) + 0.5 * HYPER.alpha * dot(mean, mean);
  return { m, fit: -eMn, occam: -0.5 * logDet(precision), total: logEvidence(design, TARGETS, HYPER) };
}

const TERMS = DEGREES.map(evidenceTermsAt);
const ALL_VALUES = TERMS.flatMap((t) => [t.fit, t.occam, t.total]);
const RANGE_LOW = Math.min(...ALL_VALUES);
const RANGE_HIGH = Math.max(...ALL_VALUES);
const PAD = (RANGE_HIGH - RANGE_LOW) * 0.08 || 1;
const Y_DOMAIN: readonly [number, number] = [RANGE_LOW - PAD, RANGE_HIGH + PAD];

export default function EvidenceTerms() {
  const tokens = useResolvedTokens();

  return (
    <div>
      <Plot
        height={220}
        xDomain={[0, 9]}
        yDomain={Y_DOMAIN}
        label="Fit term, Occam term, and total log evidence against polynomial degree M"
      >
        <Axes x={{ label: 'M', ticks: DEGREES }} y={{ label: 'log value' }} grid zeroLine />
        <Curve points={TERMS.map((t) => [t.m, t.fit] as const)} color={tokens.series[0]!} width={1.75} />
        <Curve points={TERMS.map((t) => [t.m, t.occam] as const)} color={tokens.series[1]!} width={1.75} />
        <Curve points={TERMS.map((t) => [t.m, t.total] as const)} color={tokens.series[2]!} width={2.25} />
        <Legend
          entries={[
            { label: 'fit term -E(mN)', color: tokens.series[0]!, mark: 'line' },
            { label: 'Occam term -½ln|A|', color: tokens.series[1]!, mark: 'line' },
            { label: 'total log evidence', color: tokens.series[2]!, mark: 'line' },
          ]}
          placement="bottom-right"
        />
      </Plot>
      <p className="widget-readout">
        The fit term rises as M grows and the fit improves; the Occam term falls as the model claims more
        parameters; their sum, the total log evidence, turns over rather than rising forever.
      </p>
    </div>
  );
}
