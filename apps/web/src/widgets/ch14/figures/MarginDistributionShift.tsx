import { adaBoostFit, adaBoostFunctionValue } from '@prml/math';
import { Axes, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { boostingToyData } from '../data.js';
import '../../widgets.css';

const ROUNDS = 6;
const { X, t } = boostingToyData(30);
const FIT = adaBoostFit(X, t, ROUNDS);

function margins(upToRound: number): number[] {
  return X.map((x, i) => t[i]! * adaBoostFunctionValue(FIT, x, upToRound));
}

const AFTER_ONE = margins(1);
const AFTER_ALL = margins(ROUNDS);

export default function MarginDistributionShift() {
  const tokens = useResolvedTokens();
  const maxAbs = Math.max(...AFTER_ONE.map(Math.abs), ...AFTER_ALL.map(Math.abs));

  return (
    <Plot height={200} xDomain={[-maxAbs * 1.1, maxAbs * 1.1]} yDomain={[0, 2]} label="Margin t times f(x) for each point, after one round against after all six">
      <Axes x={{ label: 'margin t f(x)' }} y={false} />
      <Rule x={0} />
      <ScatterField points={AFTER_ONE.map((m) => ({ x: m, y: 1.4, color: tokens.series[0]!, size: 3 }))} />
      <ScatterField points={AFTER_ALL.map((m) => ({ x: m, y: 0.6, color: tokens.series[1]!, size: 3 }))} />
    </Plot>
  );
}
