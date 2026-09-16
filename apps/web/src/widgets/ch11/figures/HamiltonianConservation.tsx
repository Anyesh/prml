import { hamiltonian, kineticEnergy, leapfrogTrajectory, solve } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const COV = [
  [2, 0],
  [0, 0.05],
];
function energyFn(z: readonly number[]): number {
  const solved = solve(COV, z);
  return 0.5 * (z[0]! * solved[0]! + z[1]! * solved[1]!);
}
function gradEnergyFn(z: readonly number[]): number[] {
  return solve(COV, z);
}

export default function HamiltonianConservation() {
  const tokens = useResolvedTokens();
  const z0 = [1.2, 0.1];
  const r0 = [0.3, 1.1];

  const small = leapfrogTrajectory({ z: z0, r: r0 }, gradEnergyFn, 0.05, 80);
  const large = leapfrogTrajectory({ z: z0, r: r0 }, gradEnergyFn, 0.35, 80);

  const hOf = (states: typeof small) => states.map((s, i) => [i, hamiltonian(s.z, s.r, energyFn)] as const);
  const h0 = energyFn(z0) + kineticEnergy(r0);

  return (
    <div className="widget-grid">
      <Plot width={420} height={240} xDomain={[0, 80]} yDomain={[h0 - 1.5, h0 + 1.5]} label="The Hamiltonian along a leapfrog trajectory, small step against large step">
        <Axes x={{ label: 'leapfrog step' }} y={{ label: 'H(z, r)' }} grid />
        <Curve points={hOf(small)} color={tokens.series[0]!} width={2} />
        <Curve points={hOf(large)} color={tokens.series[1]!} width={2} dash="dashed" />
      </Plot>
      <p className="widget-readout">
        H should be exactly constant along the true continuous dynamics. Leapfrog only approximates that: at epsilon
        = 0.05 it barely drifts, at epsilon = 0.35 it oscillates visibly, which is precisely the residual error the
        Metropolis accept step in 11.5.2 is there to correct.
      </p>
    </div>
  );
}
