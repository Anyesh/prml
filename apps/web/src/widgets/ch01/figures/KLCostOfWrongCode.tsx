import { useState } from 'react';
import { gaussianKlDivergence, linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

const GRID = linspace(-8, 10, 300);

export default function KLCostOfWrongCode() {
  const [muP, setMuP] = useState(0);
  const [muQ, setMuQ] = useState(2);
  const [sigmaQ, setSigmaQ] = useState(1.5);
  const tokens = useResolvedTokens();

  const p = { mu: muP, sigma2: 1 };
  const q = { mu: muQ, sigma2: sigmaQ * sigmaQ };

  const pCurve = GRID.map((x) => [x, normalPdf(x, p)] as const);
  const qCurve = GRID.map((x) => [x, normalPdf(x, q)] as const);
  const klPQ = gaussianKlDivergence(p, q);
  const klQP = gaussianKlDivergence(q, p);

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={[-8, 10]} yDomain={[0, 0.5]} label="Two Gaussians, p fixed and q draggable">
        <Axes x={{ label: 'x' }} y={{ label: 'density' }} grid />
        <Curve points={pCurve} color={tokens.series[0]!} width={2} />
        <Curve points={qCurve} color={tokens.series[1]!} width={2} />
        <Legend entries={[{ label: 'p (true)', color: tokens.series[0]!, mark: 'line' }, { label: 'q (coding distribution)', color: tokens.series[1]!, mark: 'line' }]} placement="top-right" />
      </Plot>
      <Panel columns={3} dense>
        <Slider label="mean of p" value={muP} onChange={setMuP} min={-5} max={5} step={0.1} />
        <Slider label="mean of q" value={muQ} onChange={setMuQ} min={-5} max={8} step={0.1} />
        <Slider label="std of q" value={sigmaQ} onChange={setSigmaQ} min={0.3} max={4} step={0.1} />
      </Panel>
      <p className="widget-readout">
        {`KL(p||q) = ${klPQ.toFixed(3)} nats: the extra code length paid for using q where p is true. KL(q||p) = ${klQP.toFixed(3)} nats, a different number, since coding for the wrong distribution costs differently depending on which one is wrong.`}
      </p>
    </div>
  );
}
