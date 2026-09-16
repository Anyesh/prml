import { useMemo, useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const GRID = linspace(-6, 10, 241);

export default function GaussianDraggable() {
  const [mu, setMu] = useState(2);
  const [sigma, setSigma] = useState(1.5);
  const tokens = useResolvedTokens();

  const sigma2 = sigma * sigma;
  const curve = useMemo(() => GRID.map((x) => [x, normalPdf(x, { mu, sigma2 })] as const), [mu, sigma2]);
  const peak = normalPdf(mu, { mu, sigma2 });

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-6, 10]} yDomain={[0, Math.max(0.5, peak * 1.15)]} label="Gaussian density with draggable mean and standard deviation">
        <Axes x={{ label: 'x' }} y={{ label: 'N(x | mu, sigma^2)' }} grid />
        <Curve points={curve} color={tokens.color.accent} width={2} />
        <Rule x={mu} color={tokens.color.ink} label="mean" />
        <Rule x={mu - sigma} color={tokens.color.inkFaint} label="-1 sigma" />
        <Rule x={mu + sigma} color={tokens.color.inkFaint} label="+1 sigma" />
      </Plot>
      <Panel columns={2} dense>
        <Slider label="mean, mu" value={mu} onChange={setMu} min={-4} max={8} step={0.1} />
        <Slider label="standard deviation, sigma" value={sigma} onChange={setSigma} min={0.3} max={4} step={0.1} />
        <p className="widget-readout">
          {`Peak height is 1/sqrt(2*pi*sigma^2) = ${peak.toFixed(3)}. Doubling sigma halves the peak and doubles the spread: the area underneath always stays 1.`}
        </p>
      </Panel>
    </div>
  );
}
