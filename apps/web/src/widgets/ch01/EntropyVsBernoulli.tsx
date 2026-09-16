import { useMemo, useState } from 'react';
import { discreteEntropy, linspace } from '@prml/math';
import { Axes, Curve, Rule, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const MU_GRID = linspace(0.001, 0.999, 300);
const LN2 = Math.log(2);

export default function EntropyVsBernoulli() {
  const [mu, setMu] = useState(0.5);
  const tokens = useResolvedTokens();

  const curve = useMemo(() => MU_GRID.map((m) => [m, discreteEntropy([m, 1 - m]) / LN2] as const), []);
  const hBits = discreteEntropy([mu, 1 - mu]) / LN2;

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[0, 1]} yDomain={[0, 1.05]} label="Entropy of a Bernoulli(mu) distribution, in bits">
        <Axes x={{ label: 'mu' }} y={{ label: 'H (bits)' }} grid />
        <Curve points={curve} color={tokens.color.accent} width={2.5} />
        <Rule x={0.5} color={tokens.color.inkFaint} dash label="mu = 0.5" />
        <ScatterField points={[{ x: mu, y: hBits, id: 'marker' }]} color={tokens.color.ink} size={5} />
      </Plot>
      <Slider label="mu" value={mu} onChange={setMu} min={0.001} max={0.999} step={0.001} />
      <p className="widget-readout">
        {`H(mu=${mu.toFixed(2)}) = ${hBits.toFixed(3)} bits. The peak, exactly 1 bit, sits at mu=0.5, where the two outcomes are equally likely and a coin-flip's worth of information is genuinely needed to tell them apart.`}
      </p>
    </div>
  );
}
