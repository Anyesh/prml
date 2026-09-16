import {
  evalGrid,
  linspace,
  mixtureLogisticFitEM,
  sigmoid,
  type LogisticMixtureParams,
} from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, sequentialScale, useResolvedTokens } from '@prml/viz';
import { mixtureLogisticData } from '../data.js';
import '../../widgets.css';

const { design, targets } = mixtureLogisticData(60);

const INITIAL: LogisticMixtureParams = {
  weights: [
    [0, 1, 1],
    [0, -1, -1],
  ],
  mixing: [0.5, 0.5],
};
const FIT = mixtureLogisticFitEM(design, targets, INITIAL, 20);
const MIXTURE_PARAMS = FIT.paramsHistory[FIT.paramsHistory.length - 1]!;

function mixturePosterior(params: LogisticMixtureParams, x1: number, x2: number): number {
  const phi = [1, x1, x2];
  return params.weights.reduce((sum, w, k) => {
    const y = sigmoid(w[0]! * phi[0]! + w[1]! * phi[1]! + w[2]! * phi[2]!);
    return sum + params.mixing[k]! * y;
  }, 0);
}

const AXIS = linspace(-2, 2, 50);
const GRID = evalGrid(AXIS, AXIS, (x1, x2) => mixturePosterior(MIXTURE_PARAMS, x1, x2));
const COLOR = sequentialScale([0, 1]);

export default function LogisticMixtureBoundaries() {
  const tokens = useResolvedTokens();

  return (
    <Plot width={280} height={260} xDomain={[-2, 2]} yDomain={[-2, 2]} equalAspect label="Mixture-of-two-logistic-models posterior probability, with the training points">
      <Axes x={{ label: 'x1' }} y={{ label: 'x2' }} grid />
      <ContourField data={GRID} levels={[0.2, 0.4, 0.5, 0.6, 0.8]} color={COLOR} fill={COLOR} opacity={0.5} />
      <ScatterField points={design.map((row, i) => ({ x: row[1]!, y: row[2]!, color: tokens.series[targets[i]!]!, size: 3.5 }))} />
    </Plot>
  );
}
