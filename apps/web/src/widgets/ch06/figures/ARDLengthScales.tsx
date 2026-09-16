import { fitGPRegression, gpLogMarginalLikelihood, pcg32, rbfKernel, standardNormal } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const rng = pcg32(2024);
const N = 30;
const X1 = Array.from({ length: N }, () => standardNormal(rng));
const X2 = Array.from({ length: N }, () => standardNormal(rng));
const T = X1.map((x) => Math.sin(x) + 0.1 * standardNormal(rng));
const POINTS = X1.map((x1, i) => [x1, X2[i]!]);
const CANDIDATES = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50];

let best = { l1: 1, l2: 1, value: -Infinity };
for (const l1 of CANDIDATES) {
  for (const l2 of CANDIDATES) {
    const model = fitGPRegression(rbfKernel([l1, l2]), POINTS, T, 0.05, 1e-6);
    const value = gpLogMarginalLikelihood(model);
    if (value > best.value) best = { l1, l2, value };
  }
}

export default function ARDLengthScales() {
  const tokens = useResolvedTokens();

  return (
    <div className="widget-grid">
      <Plot height={180} xDomain={[-1, 2]} yDomain={[0, best.l2 * 1.15]} label="Length scale that maximises the marginal likelihood, one bar per input">
        <Axes x={{ label: 'input', ticks: [0, 1] }} y={{ label: 'length scale (log-spaced search grid)', format: (v) => v.toFixed(0) }} grid />
        <Bars bars={[{ at: 0, value: best.l1, color: tokens.series[0]! }, { at: 1, value: best.l2, color: tokens.series[1]! }]} thickness={0.5} />
      </Plot>
      <p className="widget-readout">
        {`t depends on x1 through sin(x1); x2 is independent noise. Grid-searching (6.71)'s two length scales against the marginal likelihood picks ${best.l1} for x1 and ${best.l2} for x2, the largest value the search grid allowed: x2's coefficient wants to keep growing, which is exactly automatic relevance determination switching an input off.`}
      </p>
    </div>
  );
}
