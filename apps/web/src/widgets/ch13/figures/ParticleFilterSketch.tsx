import { mvnPdf, mvnSample, particleFilterPredict, particleFilterWeights, pcg32 } from '@prml/math';
import { Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';

const PARTICLES_N: number[][] = [-2, -1.5, -1, -0.5, 0, 0.3, 0.6, 1, 1.4, 1.8, 2.2, 2.6].map((z) => [z]);
const WEIGHTS_N = [0.02, 0.03, 0.05, 0.09, 0.14, 0.17, 0.15, 0.12, 0.09, 0.06, 0.04, 0.04];
const OBSERVATION_N1 = 1.6;
const TRANSITION_VARIANCE = 0.3;
const EMISSION_VARIANCE = 0.4;

export default function ParticleFilterSketch() {
  const tokens = useResolvedTokens();
  const rng = pcg32(20260917);

  const predicted = particleFilterPredict(rng, PARTICLES_N, WEIGHTS_N, (r, z) =>
    mvnSample(r, { mean: z as number[], cov: [[TRANSITION_VARIANCE]] }),
  );
  const rawLikelihoods = predicted.map((z) => mvnPdf([OBSERVATION_N1], { mean: [z[0]!], cov: [[EMISSION_VARIANCE]] }));
  const weightsN1 = particleFilterWeights(rawLikelihoods);

  const rowN = PARTICLES_N.map((z, l) => ({ x: z[0]!, y: 1, color: tokens.color.inkMuted, size: 3 + 26 * WEIGHTS_N[l]! }));
  const rowN1 = predicted.map((z, l) => ({ x: z[0]!, y: 0, color: tokens.color.accent, size: 3 + 26 * weightsN1[l]! }));

  return (
    <Plot width={360} height={140} xDomain={[-3, 4]} yDomain={[-0.5, 1.5]} label="Particle weights before and after one predict-reweight step">
      <Axes x={{ label: 'z' }} y={{ label: '' }} />
      <ScatterField points={rowN} label={(_, i) => `n, particle ${i + 1}`} />
      <ScatterField points={rowN1} label={(_, i) => `n+1, particle ${i + 1}`} />
    </Plot>
  );
}
