import { useMemo, useState } from 'react';
import { betaPdf, betaPosterior, linspace, type BetaParams } from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'Coin-flip posterior';
export const caption =
  'Set a prior with the sliders, then flip. Watch the posterior tighten around the true rate and the ML estimate wobble on small counts.';
export const figure = '2.3';

const X = linspace(0.001, 0.999, 200);
const TRUE_MU = 0.7;

function curveOf(p: BetaParams) {
  return X.map((x) => [x, betaPdf(x, p)] as const);
}

export default function CoinFlipPosterior() {
  const [a0, setA0] = useState(2);
  const [b0, setB0] = useState(2);
  const [heads, setHeads] = useState(0);
  const [tails, setTails] = useState(0);
  const tokens = useResolvedTokens();

  const prior = { a: a0, b: b0 };
  const posterior = useMemo(
    () => betaPosterior({ a: a0, b: b0 }, heads, tails),
    [a0, b0, heads, tails],
  );
  const n = heads + tails;
  const mlEstimate = n > 0 ? heads / n : null;

  const flip = (outcome: 'heads' | 'tails') => {
    if (outcome === 'heads') setHeads((h) => h + 1);
    else setTails((t) => t + 1);
  };

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, 1]} yDomain={[0, 6]} label="Prior and posterior over the coin's bias">
        <Axes x={{ label: 'µ' }} y={{ label: 'density' }} grid />
        <Curve points={curveOf(prior)} color={tokens.color.inkFaint} dash="dashed" width={1.5} />
        <Curve points={curveOf(posterior)} color={tokens.color.accent} width={2} />
        {mlEstimate !== null ? <Rule x={mlEstimate} color={tokens.color.danger} label="ML" /> : null}
        <Rule x={TRUE_MU} color={tokens.color.inkFaint} dash={true} />
        <Legend
          entries={[
            { label: 'prior', color: tokens.color.inkFaint, mark: 'line' },
            { label: 'posterior', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="Prior a₀" value={a0} onChange={setA0} min={0.1} max={20} scale="log" hint="Pseudo-count of heads before any data." />
        <Slider label="Prior b₀" value={b0} onChange={setB0} min={0.1} max={20} scale="log" hint="Pseudo-count of tails before any data." />
        <button type="button" className="widget-reset" onClick={() => flip('heads')}>
          Flip heads
        </button>
        <button type="button" className="widget-reset" onClick={() => flip('tails')}>
          Flip tails
        </button>
        <p className="widget-readout">
          {n === 0
            ? `${heads} heads, ${tails} tails so far. This is the prior: no data yet.`
            : `${heads} heads, ${tails} tails. Posterior is Beta(${posterior.a.toFixed(1)}, ${posterior.b.toFixed(1)}), ML estimate ${(mlEstimate! * 100).toFixed(0)}%. The coin is actually biased at ${TRUE_MU * 100}%.`}
        </p>
        <button type="button" className="widget-reset" onClick={() => { setHeads(0); setTails(0); }}>
          Reset flips
        </button>
      </Panel>
    </div>
  );
}
