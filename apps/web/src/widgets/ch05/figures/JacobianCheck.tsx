import { forwardPass, initializeWeights, pcg32, type NetworkSpec } from '@prml/math';
import { Axes, Bars, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const SPEC: NetworkSpec = { layerSizes: [3, 4, 2], hiddenActivation: 'tanh', outputActivation: 'linear' };
const WEIGHTS = initializeWeights(SPEC, pcg32(20260540), 1.1);
const INPUT = [0.4, -0.6, 0.2];
const OUTPUT_INDEX = 0;
const H = 1e-4;

function jacobianRow(): number[] {
  return INPUT.map((_, i) => {
    const plus = [...INPUT];
    const minus = [...INPUT];
    plus[i] = INPUT[i]! + H;
    minus[i] = INPUT[i]! - H;
    const yPlus = forwardPass(SPEC, WEIGHTS, plus).output[OUTPUT_INDEX]!;
    const yMinus = forwardPass(SPEC, WEIGHTS, minus).output[OUTPUT_INDEX]!;
    return (yPlus - yMinus) / (2 * H);
  });
}

const ROW = jacobianRow();

export default function JacobianCheck() {
  const tokens = useResolvedTokens();
  const bars = ROW.map((value, i) => ({ at: i, value, color: tokens.color.accent }));
  const bound = Math.max(...ROW.map(Math.abs)) * 1.2;

  return (
    <Plot height={200} xDomain={[-0.5, 2.5]} yDomain={[-bound, bound]} label="Output sensitivity to each input, ∂y₀/∂xᵢ">
      <Axes x={{ label: 'input index i' }} y={{ label: '∂y₀/∂xᵢ' }} grid zeroLine />
      <Bars bars={bars} thickness={0.5} />
    </Plot>
  );
}
