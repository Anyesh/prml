import {
  backpropGradientSingle,
  flattenWeights,
  initializeWeights,
  networkErrorSingle,
  numericalGradient,
  pcg32,
  unflattenWeights,
  type NetworkSpec,
} from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [3, 4, 2], hiddenActivation: 'tanh', outputActivation: 'linear' };
const WEIGHTS = initializeWeights(SPEC, pcg32(20260540), 1.1);
const INPUT = [0.4, -0.6, 0.2];
const TARGET = [0.3, -0.1];

const analytic = flattenWeights(backpropGradientSingle(SPEC, WEIGHTS, INPUT, TARGET, 'sumSquared'));
const numerical = numericalGradient((flat) => networkErrorSingle(SPEC, unflattenWeights(SPEC, flat), INPUT, TARGET, 'sumSquared'), flattenWeights(WEIGHTS));

const bound = Math.max(...analytic.map(Math.abs), ...numerical.map(Math.abs)) * 1.1;

export default function GradientCheckAgreement() {
  const tokens = useResolvedTokens();
  const points = analytic.map((a, i) => ({ x: a, y: numerical[i]!, id: i }));

  return (
    <Plot height={240} xDomain={[-bound, bound]} yDomain={[-bound, bound]} equalAspect label="Backpropagation's gradient against a central-difference numerical gradient, one point per weight">
      <Axes x={{ label: 'backpropagation ∂E/∂w' }} y={{ label: 'central difference ∂E/∂w' }} grid zeroLine />
      <Curve points={[[-bound, -bound], [bound, bound]]} color={tokens.color.inkFaint} width={1} dash="dashed" />
      <ScatterField points={points} color={tokens.color.accent} size={4} />
    </Plot>
  );
}
