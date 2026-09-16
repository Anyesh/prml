import { chainedPartitionFunctionRatio, interpolatedEnergy, normalSample, partitionFunctionRatioEstimate, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SIGMA_E = 1;
const N_SAMPLES = 200;
const N_REPEATS = 16;
const N_LINKS = 6;
const SIGMA_G_VALUES = [1.2, 1.6, 2, 2.5, 3, 3.5, 4];

function energyOf(sigma2: number) {
  return (z: number) => (z * z) / (2 * sigma2);
}

function directEstimate(seed: number, sigmaG: number): number {
  const rng = pcg32(seed, 210);
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
    const sigmaLo2 = 1 / (2 * energyLo(1));
    const rng = pcg32(seed * 100 + k, 211);
    const samples = Array.from({ length: N_SAMPLES }, () => normalSample(rng, { mu: 0, sigma2: sigmaLo2 }));
    ratios.push(partitionFunctionRatioEstimate(samples, energyHi, energyLo));
  }
  return chainedPartitionFunctionRatio(ratios);
}

function spread(values: number[]): number {
  return Math.max(...values) - Math.min(...values);
}

export default function SpreadVsMismatch() {
  const tokens = useResolvedTokens();
  const rows = SIGMA_G_VALUES.map((sigmaG) => {
    const direct = Array.from({ length: N_REPEATS }, (_, i) => directEstimate(3000 + i, sigmaG));
    const chained = Array.from({ length: N_REPEATS }, (_, i) => chainedEstimate(4000 + i, sigmaG));
    return { sigmaG, directSpread: spread(direct), chainedSpread: spread(chained) };
  });

  return (
    <div className="widget-grid">
      <Plot width={420} height={240} xDomain={[1, 4]} yDomain={[0, Math.max(...rows.map((r) => r.directSpread)) * 1.1]} label="Spread of the ratio estimate across repeats, as the proposal moves away from the target">
        <Axes x={{ label: 'proposal scale sigma_G' }} y={{ label: 'spread across 16 repeats' }} grid />
        <Curve points={rows.map((r) => [r.sigmaG, r.directSpread] as const)} color={tokens.series[1]!} width={2.5} />
        <Curve points={rows.map((r) => [r.sigmaG, r.chainedSpread] as const)} color={tokens.series[0]!} width={2.5} dash="dashed" />
        <Legend
          entries={[
            { label: 'direct importance sampling', color: tokens.series[1]!, mark: 'line' },
            { label: 'chained through 6 links', color: tokens.series[0]!, mark: 'dashed-line' },
          ]}
        />
      </Plot>
      <p className="widget-readout">
        The direct estimate's spread grows with every step the proposal takes away from the target; the chained
        estimate's spread grows far more slowly, because no single link in the chain is ever badly mismatched.
      </p>
    </div>
  );
}
