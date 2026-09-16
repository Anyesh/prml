import { useMemo, useState } from 'react';
import {
  designMatrix,
  eigSym,
  linspace,
  matmul,
  maximiseEvidence,
  pcg32,
  polynomialBasis,
  standardNormal,
  transpose,
} from '@prml/math';
import { Annotation, Axes, Curve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const DATASET_SEED = 20260916;
const DATASET_SIZE = 25;
const NOISE_STD = 0.2;
const ALPHA_LOG_DOMAIN: readonly [number, number] = [-4, 4];
const ALPHA_LOG_STEPS = 100;
const ALPHA_TICKS = [-4, -2, 0, 2, 4];
const INITIAL_HYPER = { alpha: 1e-2, beta: 1 };
const DEFAULT_DEGREE = 9;

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
const ALPHA_LOG = linspace(ALPHA_LOG_DOMAIN[0], ALPHA_LOG_DOMAIN[1], ALPHA_LOG_STEPS);

export default function GammaAgainstAlpha() {
  const [degree, setDegree] = useState(DEFAULT_DEGREE);
  const tokens = useResolvedTokens();

  const design = useMemo(() => designMatrix(XS, polynomialBasis(degree)), [degree]);
  const nominalParameters = degree + 1;

  // The joint fixed point re-estimates both alpha and beta together; beta is held at that
  // converged value here so the swept curve passes exactly through the point the real
  // iteration lands on.
  const converged = useMemo(() => maximiseEvidence(design, TARGETS, INITIAL_HYPER), [design]);

  const gramEigenvalues = useMemo(
    () => eigSym(matmul(transpose(design), design)).values.map((e) => converged.beta * e),
    [design, converged.beta],
  );

  const curve = ALPHA_LOG.map((logAlpha) => {
    const alpha = 10 ** logAlpha;
    const gamma = gramEigenvalues.reduce((s, lam) => s + lam / (alpha + lam), 0);
    return [logAlpha, gamma] as const;
  });

  const convergedLogAlpha = Math.log10(converged.alpha);

  return (
    <div className="widget-grid">
      <Plot
        height={220}
        xDomain={ALPHA_LOG_DOMAIN}
        yDomain={[0, nominalParameters + 1]}
        label="Effective number of parameters gamma against alpha, PRML 3.91"
      >
        <Axes x={{ label: 'α', ticks: ALPHA_TICKS, format: (v) => `1e${Math.round(v)}` }} y={{ label: 'γ' }} grid />
        <Rule y={nominalParameters} color={tokens.color.inkFaint} label={`M + 1 = ${nominalParameters}`} />
        <Curve points={curve} color={tokens.color.accent} width={2} />
        <ScatterField
          points={[{ x: convergedLogAlpha, y: converged.effectiveParameters, color: tokens.color.danger, size: 5 }]}
          label={() => 'evidence maximisation settles here'}
        />
        <Annotation
          x={convergedLogAlpha}
          y={converged.effectiveParameters}
          text={`α=${converged.alpha.toExponential(1)}, γ=${converged.effectiveParameters.toFixed(2)}`}
          color={tokens.color.danger}
          anchor="start"
          dx={8}
          dy={-8}
          plate
        />
      </Plot>
      <Panel dense>
        <Slider
          label="Polynomial degree M"
          value={degree}
          onChange={(v) => setDegree(Math.round(v))}
          min={0}
          max={9}
          step={1}
          hint="Sets the basis and the M+1 ceiling that gamma approaches as alpha shrinks toward zero."
        />
      </Panel>
    </div>
  );
}
