import { useMemo, useState } from 'react';
import { autocorrelation, gaussianRandomWalkProposal, metropolisHastingsChain, mvnLogPdf, pcg32 } from '@prml/math';
import { Axes, CovarianceEllipse, Curve, Plot, ScatterField, Trajectory, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const MEAN = [0, 0];
const COV = [
  [1, 0.9],
  [0.9, 1],
];
const N_STEPS = 300;
const MAX_LAG = 40;


function circlePoints(cx: number, cy: number, r: number, n = 48): (readonly [number, number])[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    return [cx + r * Math.cos(t), cy + r * Math.sin(t)] as const;
  });
}

export default function MetropolisHastingsExplorer() {
  const [logStepSize, setLogStepSize] = useState(-0.4);
  const tokens = useResolvedTokens();
  const stepSize = 10 ** logStepSize;

  const chain = useMemo(() => {
    const rng = pcg32(2026, 50);
    const proposal = gaussianRandomWalkProposal(stepSize);
    return metropolisHastingsChain(rng, [0, 0], N_STEPS, { ...proposal, targetLogPdf: (z) => mvnLogPdf(z, { mean: MEAN, cov: COV }) });
  }, [stepSize]);

  const rho = useMemo(() => autocorrelation(chain.states.map((s) => s[0]!), MAX_LAG), [chain]);
  const current = chain.states[chain.states.length - 1]!;

  return (
    <div className="widget-grid">
      <Plot width={340} height={340} xDomain={[-4, 4]} yDomain={[-4, 4]} equalAspect label="Metropolis-Hastings chain on a correlated Gaussian target">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <CovarianceEllipse mean={MEAN} cov={COV} levels={[0.9]} color={tokens.color.inkFaint} width={1.5} />
        <Trajectory path={chain.states.map((s) => [s[0]!, s[1]!] as const)} color={tokens.series[0]!} width={1.25} fadeOlder />
        <Curve points={circlePoints(current[0]!, current[1]!, stepSize)} color={tokens.color.accent} width={1.5} dash="dashed" />
        <ScatterField points={[{ x: current[0]!, y: current[1]!, color: tokens.color.accent, size: 5 }]} />
      </Plot>
      <Plot width={340} height={200} xDomain={[0, MAX_LAG]} yDomain={[-0.2, 1]} label="Autocorrelation of the first coordinate against lag">
        <Axes x={{ label: 'lag' }} y={{ label: 'autocorrelation' }} grid zeroLine />
        <Curve points={rho.map((r, lag) => [lag, r] as const)} color={tokens.series[1]!} width={2} />
      </Plot>
      <Slider
        label="Proposal step size (log scale)"
        value={logStepSize}
        onChange={setLogStepSize}
        min={-2}
        max={1}
        step={0.02}
        format={() => stepSize.toFixed(3)}
        hint="Tiny: high acceptance, the trace barely moves. Huge: the dashed circle dwarfs the ellipse and almost every proposal is rejected."
      />
      <p className="widget-readout">
        Step size {stepSize.toFixed(3)}: acceptance rate {(chain.acceptanceRate * 100).toFixed(1)}%, autocorrelation still{' '}
        {(rho[MAX_LAG]! * 100).toFixed(1)}% at lag {MAX_LAG}.
      </p>
    </div>
  );
}
