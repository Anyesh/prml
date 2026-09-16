import { useState } from 'react';
import { dot, mixtureExpertsFitEM, softmax, type ExpertsParams } from '@prml/math';
import { Axes, FunctionCurve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { expertsRegimeChangeData } from './data.js';
import '../widgets.css';

export const title = 'A gate learning three regimes at once';
export const caption =
  'Step through EM iterations. The left panel is the data with each expert\'s current line; the right panel is the gate\'s current probability of each expert, one curve per colour, which sharpen into a soft partition with two boundaries.';
export const figure = '14.5.3';

const ITERATIONS = 12;
const DOMAIN: readonly [number, number] = [-3, 3];
const { design, t } = expertsRegimeChangeData(60);

const INITIAL_PARAMS: ExpertsParams = {
  expertWeights: [
    [0, 1],
    [0, -1],
    [0, 1],
  ],
  beta: 5,
  gateWeights: [
    [0, 0],
    [0.5, 1],
    [0.5, -1],
  ],
};

const FIT = mixtureExpertsFitEM(design, design, t, INITIAL_PARAMS, ITERATIONS);

export default function MixtureOfExpertsRegimeChange() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const params = FIT.paramsHistory[step]!;
  const numExperts = params.expertWeights.length;

  const gateProbability = (k: number) => (x: number) => {
    const activations = params.gateWeights.map((w) => dot(w, [1, x]));
    return softmax(activations)[k]!;
  };
  const expertLine = (k: number) => (x: number) => params.expertWeights[k]![0]! + params.expertWeights[k]![1]! * x;

  const legendEntries = Array.from({ length: numExperts }, (_, k) => ({
    label: `expert ${k + 1}`,
    color: tokens.series[k]!,
    mark: 'line' as const,
  }));

  return (
    <div>
      <div className="widget-grid">
        <Plot height={260} xDomain={DOMAIN} yDomain={[-4, 4]} label="Data with three genuine regimes, and each expert's current fitted line">
          <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
          <ScatterField points={design.map((row, i) => ({ x: row[1]!, y: t[i]!, color: tokens.color.inkFaint, size: 3 }))} />
          {Array.from({ length: numExperts }, (_, k) => (
            <FunctionCurve key={k} f={expertLine(k)} domain={DOMAIN} color={tokens.series[k]!} width={2} />
          ))}
          <Legend entries={legendEntries} placement="bottom-right" />
        </Plot>
        <Plot height={260} xDomain={DOMAIN} yDomain={[0, 1]} label="The gate's probability of each expert across input space">
          <Axes x={{ label: 'x' }} y={{ label: 'gate probability' }} grid />
          {Array.from({ length: numExperts }, (_, k) => (
            <FunctionCurve key={k} f={gateProbability(k)} domain={DOMAIN} color={tokens.series[k]!} width={2.5} />
          ))}
        </Plot>
      </div>
      <StepThrough
        step={step}
        stepCount={FIT.paramsHistory.length}
        onStep={setStep}
        labels={FIT.paramsHistory.map((_, i) => (i === 0 ? 'initial guess' : `after EM round ${i}`))}
      />
      <p className="widget-readout">
        {step > 0
          ? `Log-likelihood before this round's M step: ${FIT.logLikelihoodHistory[step - 1]!.toFixed(2)}.`
          : 'All three experts start close to guesses; the gate starts almost flat across the three regions.'}
      </p>
    </div>
  );
}
