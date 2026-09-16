import { useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../../widgets.css';

// Fixed rather than derived from a dataset: the point of 3.70 is that the data sets the
// posterior width regardless of the prior, so this widget holds it constant and only
// lets the reader vary the prior.
const POSTERIOR_STD = 0.3;
const POSTERIOR = { mu: 0, sigma2: POSTERIOR_STD * POSTERIOR_STD };
const POSTERIOR_PEAK = normalPdf(0, POSTERIOR);
const MIN_PRIOR_WIDTH = 1;
const MAX_PRIOR_WIDTH = 30;
const SAMPLES = 160;
const PAD = 1;

export default function OccamFactor() {
  const [priorWidth, setPriorWidth] = useState(6);
  const tokens = useResolvedTokens();

  const half = priorWidth / 2;
  const domain: readonly [number, number] = [-half - PAD, half + PAD];
  const priorHeight = 1 / priorWidth;
  const priorCurve = [
    [domain[0], 0],
    [-half, 0],
    [-half, priorHeight],
    [half, priorHeight],
    [half, 0],
    [domain[1], 0],
  ] as const;
  const posteriorCurve = linspace(domain[0], domain[1], SAMPLES).map((x) => [x, normalPdf(x, POSTERIOR)] as const);
  // PRML 3.70: the Occam factor is the ratio of the posterior width the data leaves to
  // the prior width the model claimed, so a wider prior is charged more per parameter.
  const occamFactor = (2 * POSTERIOR_STD) / priorWidth;

  return (
    <div>
      <Plot
        height={220}
        xDomain={domain}
        yDomain={[0, POSTERIOR_PEAK * 1.15]}
        label="Flat prior and narrow posterior peak, illustrating the Occam factor of 3.70"
      >
        <Axes x={{ label: 'w' }} y={false} />
        <Curve points={priorCurve} color={tokens.series[0]!} width={2} />
        <Curve points={posteriorCurve} color={tokens.series[1]!} width={2} />
        <Legend
          entries={[
            { label: 'prior (flat)', color: tokens.series[0]!, mark: 'line' },
            { label: 'posterior', color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <Panel columns={2} dense>
        <Slider
          label="Prior width Δw_prior"
          value={priorWidth}
          onChange={setPriorWidth}
          min={MIN_PRIOR_WIDTH}
          max={MAX_PRIOR_WIDTH}
          scale="log"
          hint="A wider, more uncommitted prior is charged more for every parameter it adds."
        />
        <p className="widget-readout">
          {`Δw_posterior ≈ ${(2 * POSTERIOR_STD).toFixed(2)}, Δw_prior = ${priorWidth.toFixed(1)}. `}
          {`Occam factor Δw_posterior / Δw_prior ≈ ${occamFactor.toFixed(3)}.`}
        </p>
      </Panel>
    </div>
  );
}
