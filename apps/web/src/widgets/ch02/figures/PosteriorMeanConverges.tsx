import { betaPosterior, pcg32 } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const TRUE_MU = 0.7;
const N = 80;

function posteriorMeanTrace(a0: number, b0: number, seed: number) {
  const rng = pcg32(seed, 1);
  let heads = 0;
  let tails = 0;
  const trace: (readonly [number, number])[] = [];
  for (let n = 1; n <= N; n++) {
    if (rng.next() < TRUE_MU) heads++;
    else tails++;
    const post = betaPosterior({ a: a0, b: b0 }, heads, tails);
    trace.push([n, post.a / (post.a + post.b)]);
  }
  return trace;
}

// Same coin-flip stream for both priors, so the only thing that differs between the two
// curves is how much the prior resists it, not which flips happened to land.
const SEED = 20260916;
const WEAK = posteriorMeanTrace(0.5, 0.5, SEED);
const STRONG = posteriorMeanTrace(20, 20, SEED);

export default function PosteriorMeanConverges() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[1, N]} yDomain={[0, 1]} label="Posterior mean against number of flips, for a weak and a strong prior">
        <Axes x={{ label: 'flips' }} y={{ label: 'posterior mean of µ' }} grid />
        <Rule y={TRUE_MU} color={tokens.color.inkFaint} />
        <Curve points={WEAK} color={tokens.series[0]!} width={1.75} />
        <Curve points={STRONG} color={tokens.series[1]!} width={1.75} />
        <Legend
          entries={[
            { label: 'a₀=b₀=0.5', color: tokens.series[0]!, mark: 'line' },
            { label: 'a₀=b₀=20', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="bottom-right"
        />
      </Plot>
      <p className="widget-readout">
        {`Same 80 flips of a coin biased at ${TRUE_MU}, fed to two priors. The weak prior tracks the data almost `}
        {'immediately; the strong one, worth 40 fictitious flips by itself, takes most of the run to let go of 0.5.'}
      </p>
    </div>
  );
}
