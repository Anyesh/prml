import { gaussianRandomWalkProposal, metropolisHastingsChain, mvnCovarianceEllipse, mvnLogPdf, pcg32 } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, Trajectory, useResolvedTokens } from '@prml/viz';
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

function ellipsePoints(cx: number, cy: number, rx: number, ry: number, angle: number, n = 64): (readonly [number, number])[] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const t = (2 * Math.PI * i) / n;
    const x = rx * Math.cos(t);
    const y = ry * Math.sin(t);
    return [cx + x * Math.cos(angle) - y * Math.sin(angle), cy + x * Math.sin(angle) + y * Math.cos(angle)] as const;
  });
}

export default function EquilibriumFromManyStarts() {
  const tokens = useResolvedTokens();
  const ellipse = mvnCovarianceEllipse({ mean: MEAN, cov: COV }, 0.9);
  const chains = STARTS.map((start, i) => {
    const rng = pcg32(500 + i, 70);
    const proposal = gaussianRandomWalkProposal(0.9);
    return metropolisHastingsChain(rng, start, N_STEPS, { ...proposal, targetLogPdf: (z) => mvnLogPdf(z, { mean: MEAN, cov: COV }) });
  });

  return (
    <div className="widget-grid">
      <Plot width={420} height={340} xDomain={[-10, 10]} yDomain={[-10, 10]} equalAspect label="Six chains starting far apart, all converging on the same equilibrium distribution">
        <Axes x={{ label: 'z1' }} y={{ label: 'z2' }} grid zeroLine />
        <Curve points={ellipsePoints(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle)} color={tokens.color.inkFaint} width={1.5} />
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
