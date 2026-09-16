import { pcg32 } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N_STEPS = 400;
const N_CHAINS = 30;

function simulateWalk(seed: number): number[] {
  const rng = pcg32(seed, 60);
  let z = 0;
  const path = [0];
  for (let t = 0; t < N_STEPS; t++) {
    const u = rng.next();
    z += u < 0.25 ? 1 : u < 0.5 ? -1 : 0;
    path.push(z);
  }
  return path;
}

export default function RandomWalkDistance() {
  const tokens = useResolvedTokens();
  const walks = Array.from({ length: N_CHAINS }, (_, i) => simulateWalk(1000 + i));
  const meanSquare = Array.from({ length: N_STEPS + 1 }, (_, t) => walks.reduce((s, w) => s + w[t]! ** 2, 0) / N_CHAINS);
  const analytic = Array.from({ length: N_STEPS + 1 }, (_, t) => t / 2);

  return (
    <div className="widget-grid">
      <Plot width={420} height={240} xDomain={[0, N_STEPS]} yDomain={[0, N_STEPS / 2 + 20]} label="Mean squared displacement of a simple random walk, simulated against the analytic tau/2">
        <Axes x={{ label: 'step tau' }} y={{ label: 'E[(z^tau)^2]' }} grid />
        <Curve points={analytic.map((v, t) => [t, v] as const)} color={tokens.color.inkMuted} width={1.5} dash="dashed" />
        <Curve points={meanSquare.map((v, t) => [t, v] as const)} color={tokens.series[0]!} width={2} />
      </Plot>
      <p className="widget-readout">
        {N_CHAINS} independent walks averaged against PRML 11.34-11.36's exact result,
        E[(z^tau)^2] = tau/2: distance travelled grows as sqrt(tau), the signature of every random-walk proposal in this chapter.
      </p>
    </div>
  );
}
