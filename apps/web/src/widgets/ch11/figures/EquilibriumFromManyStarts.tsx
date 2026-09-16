import { gaussianRandomWalkProposal, metropolisHastingsChain, mvnLogPdf, pcg32 } from '@prml/math';
import { Axes, CovarianceEllipse, Plot, ScatterField, Trajectory, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const MEAN = [0, 0];
const COV = [
  [1, 0.9],
  [0.9, 1],
];
const STARTS: [number, number][] = [
  [8, 8],
  [-8, 8],
  [8, -8],
  [-8, -8],
  [0, 9],
  [0, -9],
];
const N_STEPS = 70;


export default function EquilibriumFromManyStarts() {
  const tokens = useResolvedTokens();
  const chains = STARTS.map((start, i) => {
    const rng = pcg32(500 + i, 70);
    const proposal = gaussianRandomWalkProposal(0.9);
    return metropolisHastingsChain(rng, start, N_STEPS, { ...proposal, targetLogPdf: (z) => mvnLogPdf(z, { mean: MEAN, cov: COV }) });
  });

  return (
    <div className="widget-grid">
      <Plot width={420} height={340} xDomain={[-10, 10]} yDomain={[-10, 10]} equalAspect label="Six chains starting far apart, all converging on the same equilibrium distribution">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <CovarianceEllipse mean={MEAN} cov={COV} levels={[0.9]} color={tokens.color.inkFaint} width={1.5} />
        {chains.map((c, i) => (
          <Trajectory key={i} path={c.states.map((s) => [s[0]!, s[1]!] as const)} color={tokens.series[i % tokens.series.length]!} width={1} fadeOlder />
        ))}
        <ScatterField
          points={chains.map((c, i) => {
            const last = c.states[c.states.length - 1]!;
            return { x: last[0]!, y: last[1]!, color: tokens.series[i % tokens.series.length]!, size: 5 };
          })}
        />
      </Plot>
      <p className="widget-readout">
        Every chain starts at a different corner and none start anywhere near the target; after {N_STEPS} steps all six
        have forgotten where they began (PRML 11.39-11.41's ergodicity argument, made visible).
      </p>
    </div>
  );
}
