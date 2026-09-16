import { useMemo, useState } from 'react';
import { gaussianRandomWalkProposal, hmcChain, metropolisHastingsChain, mvnCovarianceEllipse, mvnLogPdf, pcg32, solve } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, Trajectory, useResolvedTokens } from '@prml/viz';
import { Slider, StepThrough } from '@prml/ui';
import '../widgets.css';

const MEAN = [0, 0];
const COV = [
  [2, 0],
  [0, 0.05],
];
const N_ITERS = 25;

function energyFn(z: readonly number[]): number {
  return -mvnLogPdf(z, { mean: MEAN, cov: COV });
}
function gradEnergyFn(z: readonly number[]): number[] {
  return solve(COV, [z[0]! - MEAN[0]!, z[1]! - MEAN[1]!]);
}

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number, n = 64): (readonly [number, number])[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

export default function HmcVsRandomWalk() {
  const [epsilon, setEpsilon] = useState(0.15);
  const [l, setL] = useState(15);
  const [iter, setIter] = useState(N_ITERS - 1);
  const tokens = useResolvedTokens();

  const hmc = useMemo(() => hmcChain(pcg32(2026, 150), [0, 0], N_ITERS, { energyFn, gradEnergyFn, epsilon, l }), [epsilon, l]);
  const walk = useMemo(() => {
    const proposal = gaussianRandomWalkProposal(epsilon);
    return metropolisHastingsChain(pcg32(2026, 151), [0, 0], N_ITERS * l, { ...proposal, targetLogPdf: (z) => mvnLogPdf(z, { mean: MEAN, cov: COV }) });
  }, [epsilon, l]);

  const ellipse = mvnCovarianceEllipse({ mean: MEAN, cov: COV }, 0.9);
  const trajectory = hmc.trajectories[iter]!.map((s) => [s.z[0]!, s.z[1]!] as const);
  const hmcHistory = hmc.states.slice(0, iter + 1).map((s) => [s[0]!, s[1]!] as const);
  const walkHistory = walk.states.slice(0, (iter + 1) * l).map((s) => [s[0]!, s[1]!] as const);
  const hmcAccepted = hmc.accepted[iter]!;

  const hmcDisplacement = Math.hypot(hmcHistory[hmcHistory.length - 1]![0] - 0, hmcHistory[hmcHistory.length - 1]![1] - 0);
  const walkDisplacement = Math.hypot(walkHistory[walkHistory.length - 1]![0] - 0, walkHistory[walkHistory.length - 1]![1] - 0);

  return (
    <div className="widget-grid">
      <Plot width={380} height={340} xDomain={[-4, 4]} yDomain={[-4, 4]} equalAspect label="Hybrid Monte Carlo: the current leapfrog trajectory before accept/reject">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <Curve points={ellipsePoints(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle)} color={tokens.color.inkFaint} width={1.5} />
        <Trajectory path={hmcHistory} color={tokens.color.inkMuted} width={1} fadeOlder />
        <Curve points={trajectory} color={hmcAccepted ? tokens.color.success : tokens.color.danger} width={2} />
        <ScatterField points={[{ x: hmcHistory[hmcHistory.length - 1]![0], y: hmcHistory[hmcHistory.length - 1]![1], color: tokens.color.accent, size: 5 }]} />
      </Plot>
      <Plot width={380} height={340} xDomain={[-4, 4]} yDomain={[-4, 4]} equalAspect label="Random-walk Metropolis with the same number of target evaluations">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <Curve points={ellipsePoints(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle)} color={tokens.color.inkFaint} width={1.5} />
        <Trajectory path={walkHistory} color={tokens.series[1]!} width={1} fadeOlder />
      </Plot>
      <StepThrough step={iter} stepCount={N_ITERS} onStep={setIter} labels={hmc.accepted.map((a) => (a ? 'accepted' : 'rejected'))} />
      <Slider label="Leapfrog step size epsilon" value={epsilon} onChange={setEpsilon} min={0.02} max={0.6} step={0.01} hint="Also sets the random-walk's step size, so both spend the same evaluation budget per move." />
      <Slider label="Leapfrog steps L" value={l} onChange={(v) => setL(Math.round(v))} min={2} max={40} step={1} />
      <p className="widget-readout">
        After {iter + 1} HMC iterations ({(iter + 1) * l} gradient evaluations, acceptance{' '}
        {((hmc.accepted.slice(0, iter + 1).filter(Boolean).length / (iter + 1)) * 100).toFixed(0)}%): HMC has moved{' '}
        {hmcDisplacement.toFixed(2)} from the origin; the random walk with the same evaluation count has moved{' '}
        {walkDisplacement.toFixed(2)}.
      </p>
    </div>
  );
}
