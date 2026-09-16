import { useMemo, useState } from 'react';
import { chainedPartitionFunctionRatio, interpolatedEnergy, normalSample, partitionFunctionRatioEstimate, pcg32 } from '@prml/math';
import { Axes, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const SIGMA_E = 1;
const N_SAMPLES = 300;
const N_REPEATS = 24;
const N_LINKS = 6;

function energyOf(sigma2: number): (z: number) => number {
  return (z) => (z * z) / (2 * sigma2);
}

/** A fixed, index-keyed horizontal spread for the two repeat-estimate columns, so points do not overdraw a single vertical line. */
function jitter(i: number): number {
  return (((i * 2654435761) % 1000) / 1000 - 0.5) * 0.3;
}

function directEstimate(seed: number, sigmaG: number): number {
  const rng = pcg32(seed, 200);
  const samples = Array.from({ length: N_SAMPLES }, () => normalSample(rng, { mu: 0, sigma2: sigmaG * sigmaG }));
  return partitionFunctionRatioEstimate(samples, energyOf(SIGMA_E * SIGMA_E), energyOf(sigmaG * sigmaG));
}

function chainedEstimate(seed: number, sigmaG: number): number {
  const eG = energyOf(sigmaG * sigmaG);
  const eE = energyOf(SIGMA_E * SIGMA_E);
  const ratios: number[] = [];
  for (let k = 1; k <= N_LINKS; k++) {
    const alphaLo = (k - 1) / N_LINKS;
    const alphaHi = k / N_LINKS;
    const energyLo = interpolatedEnergy(alphaLo, eG, eE);
    const energyHi = interpolatedEnergy(alphaHi, eG, eE);
    const sigmaLo2 = 1 / (1 / (sigmaG * sigmaG) + alphaLo * (1 / (SIGMA_E * SIGMA_E) - 1 / (sigmaG * sigmaG)));
    const rng = pcg32(seed * 100 + k, 201);
    const samples = Array.from({ length: N_SAMPLES }, () => normalSample(rng, { mu: 0, sigma2: Math.max(sigmaLo2, 1e-6) }));
    ratios.push(partitionFunctionRatioEstimate(samples, energyHi, energyLo));
  }
  return chainedPartitionFunctionRatio(ratios);
}

export default function PartitionFunctionChaining() {
  const [sigmaG, setSigmaG] = useState(2.5);
  const tokens = useResolvedTokens();
  const trueRatio = SIGMA_E / sigmaG;

  const { direct, chained } = useMemo(() => {
    const d = Array.from({ length: N_REPEATS }, (_, i) => directEstimate(1000 + i, sigmaG));
    const c = Array.from({ length: N_REPEATS }, (_, i) => chainedEstimate(2000 + i, sigmaG));
    return { direct: d, chained: c };
  }, [sigmaG]);

  return (
    <div className="widget-grid">
      <Plot width={380} height={280} xDomain={[-0.6, 1.6]} yDomain={[0, Math.max(trueRatio * 3, 1)]} label="Repeated ratio estimates, direct importance sampling against chaining through intermediate distributions">
        <Axes x={{ ticks: [0, 1], format: (v) => (v === 0 ? 'direct' : 'chained') }} y={{ label: 'estimated Z_E / Z_G' }} grid />
        <Rule y={trueRatio} color={tokens.color.accent} label="true ratio" />
        <ScatterField points={direct.map((v, i) => ({ x: 0 + jitter(i), y: v, color: tokens.series[1]!, size: 3 }))} />
        <ScatterField points={chained.map((v, i) => ({ x: 1 + jitter(i), y: v, color: tokens.series[0]!, size: 3 }))} />
      </Plot>
      <Slider
        label="Proposal scale sigma_G (target sigma_E = 1)"
        value={sigmaG}
        onChange={setSigmaG}
        min={1.1}
        max={4}
        step={0.05}
        hint="Push sigma_G away from 1 and watch the direct estimates spread out while the chained ones stay put."
      />
      <p className="widget-readout">
        True ratio {trueRatio.toFixed(3)}. At sigma_G = {sigmaG.toFixed(2)}, {N_REPEATS} repeats of the direct
        estimate spread from {Math.min(...direct).toFixed(3)} to {Math.max(...direct).toFixed(3)}; chaining through{' '}
        {N_LINKS} intermediate distributions narrows that to {Math.min(...chained).toFixed(3)}-{Math.max(...chained).toFixed(3)}.
      </p>
    </div>
  );
}
