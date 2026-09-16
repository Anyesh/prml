import { useMemo, useState } from 'react';
import { adaBoostFit, adaBoostFunctionValue, evalGrid, linspace } from '@prml/math';
import { Axes, ContourField, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import { boostingToyData } from './data.js';
import '../widgets.css';

export const title = 'One AdaBoost round per click';
export const caption =
  'Step forward one round at a time. Point size is that round\'s data weight, the dashed line is the new stump, and the solid curve is every round combined so far.';
export const figure = '14.2';

const ROUNDS = 6;
const DOMAIN: readonly [number, number] = [-2.2, 2.2];
const GRID_AXIS = linspace(DOMAIN[0], DOMAIN[1], 70);
const { X, t } = boostingToyData(30);
const FIT = adaBoostFit(X, t, ROUNDS);

export default function AdaBoostStepThrough() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const grid = useMemo(
    () => evalGrid(GRID_AXIS, GRID_AXIS, (x1, x2) => adaBoostFunctionValue(FIT, [x1, x2], step)),
    [step],
  );

  const round = step > 0 ? FIT.rounds[step - 1] : undefined;
  const sizingWeights = round ? round.weights : new Array(X.length).fill(1 / X.length);
  const maxWeight = Math.max(...sizingWeights);

  const labels = ['before round 1', ...FIT.rounds.map((_, i) => `after round ${i + 1}`)];

  return (
    <div>
      <Plot width={420} height={420} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Toy classification data, weighted by the round's data weights, with the newest stump and the combined boundary">
        <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid zeroLine />
        {step > 0 ? <ContourField data={grid} levels={[0]} color={tokens.color.accent} lineWidth={2} /> : null}
        {round
          ? round.stump.featureIndex === 0
            ? <Rule x={round.stump.threshold} color={tokens.color.borderStrong} />
            : <Rule y={round.stump.threshold} color={tokens.color.borderStrong} />
          : null}
        <ScatterField
          points={X.map((x, i) => ({
            x: x[0]!,
            y: x[1]!,
            color: tokens.series[t[i]! > 0 ? 0 : 1]!,
            size: 2.5 + 14 * (sizingWeights[i]! / maxWeight),
          }))}
        />
      </Plot>
      <StepThrough step={step} stepCount={ROUNDS + 1} onStep={setStep} labels={labels} />
      <p className="widget-readout">
        {round
          ? `Round ${step}: stump on x${round.stump.featureIndex + 1} at ${round.stump.threshold.toFixed(2)}, weighted error epsilon = ${round.epsilon.toFixed(3)}, vote weight alpha = ${round.alpha.toFixed(3)}.`
          : 'Every point starts with equal weight, so round 1 trains like an ordinary single stump.'}
      </p>
    </div>
  );
}
