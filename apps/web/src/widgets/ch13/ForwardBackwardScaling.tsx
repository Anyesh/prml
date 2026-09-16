import { useMemo, useState } from 'react';
import {
  hmmBackwardScaled,
  hmmForwardScaled,
  hmmForwardUnscaled,
  hmmGaussianEmissionMatrix,
} from '@prml/math';
import { Axes, Bars, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { hmmDemo, hmmUnderflowDemo } from './data.js';

import '../widgets.css';

export const title = 'Forward-backward, scaled';
export const caption =
  'Step through the chain. Left: the scaled alpha and beta messages at this step, both genuine distributions. Right: what the unscaled alpha would have done over a much longer run.';
export const figure = '13.12';

const SHORT = hmmDemo(12);
const LONG = hmmUnderflowDemo();

export default function ForwardBackwardScaling() {
  const tokens = useResolvedTokens();
  const [step, setStep] = useState(0);

  const { alphaHat, betaHat, c } = useMemo(() => {
    const B = hmmGaussianEmissionMatrix(SHORT.data, SHORT.components);
    const forward = hmmForwardScaled(SHORT.pi, SHORT.A, B);
    const backward = hmmBackwardScaled(SHORT.A, B, forward.c);
    return { alphaHat: forward.alphaHat, betaHat: backward, c: forward.c };
  }, []);

  const unscaledSums = useMemo(() => {
    const B = hmmGaussianEmissionMatrix(LONG.data, LONG.components);
    const alpha = hmmForwardUnscaled(LONG.pi, LONG.A, B);
    return alpha.map((row) => row.reduce((a, b) => a + b, 0));
  }, []);

  const collapseStep = unscaledSums.findIndex((v) => v === 0);
  const K = SHORT.components.length;
  const N = SHORT.data.length;

  const alphaBars = alphaHat[step]!.map((v, k) => ({ at: k, value: v, color: tokens.series[k]! }));
  const betaBars = betaHat[step]!.map((v, k) => ({ at: k, value: v, color: tokens.series[k]! }));
  const maxBeta = Math.max(...betaHat.flat(), 1);

  const collapseCurve: (readonly [number, number])[] = unscaledSums.map((v, n) => [n, Math.log10(Math.max(v, 1e-320))]);

  return (
    <div>
      <div className="widget-grid">
        <div>
          <Plot width={220} height={180} xDomain={[-0.6, K - 0.4]} yDomain={[0, 1]} label="alpha-hat at this step, a distribution over states">
            <Axes x={{ label: 'state' }} y={{ label: 'alpha-hat' }} grid />
            <Bars bars={alphaBars} thickness={0.6} />
          </Plot>
          <Plot width={220} height={180} xDomain={[-0.6, K - 0.4]} yDomain={[0, maxBeta * 1.05]} label="beta-hat at this step">
            <Axes x={{ label: 'state' }} y={{ label: 'beta-hat' }} grid />
            <Bars bars={betaBars} thickness={0.6} />
          </Plot>
        </div>
        <Plot
          width={280}
          height={280}
          xDomain={[0, LONG.data.length - 1]}
          yDomain={[-320, 5]}
          label="log10 of the unscaled alpha row sum across a long chain"
        >
          <Axes x={{ label: 'step' }} y={{ label: 'log10(sum alpha)' }} grid />
          <Curve points={collapseCurve} color={tokens.color.accent} width={2} />
        </Plot>
      </div>
      <StepThrough
        step={step}
        stepCount={N}
        onStep={setStep}
        labels={Array.from({ length: N }, (_, t) => `Step ${t + 1}`)}
      />
      <p className="widget-readout">
        Step {step + 1}: scaling factor c = {c[step]!.toExponential(3)}. On the long,
        wide-variance chain at right, the unscaled alpha row sum reaches exactly 0 at step{' '}
        {collapseStep + 1} of {LONG.data.length}.
      </p>
    </div>
  );
}
