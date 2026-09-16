import { irlsFit, logisticPredict } from '@prml/math';
import { Axes, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const XS = [-3, -2.4, -1.8, -1, 0.5, 1.2, 1.8, 2.4, 3, -0.2];
const TS = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
const DESIGN = XS.map((x) => [1, x]);
const FIT = irlsFit(DESIGN, TS);

export default function ResidualInterpretation() {
  const tokens = useResolvedTokens();
  const residuals = XS.map((_x, i) => logisticPredict(FIT.weights, DESIGN[i]!) - TS[i]!);

  return (
    <Plot height={220} xDomain={[-4, 4]} yDomain={[-1.1, 1.1]} label="The gradient's per-point term, y − t, after fitting">
      <Axes x={{ label: 'x' }} y={{ label: 'y(x) − t' }} grid zeroLine />
      <Rule y={0} color={tokens.color.inkFaint} />
      <ScatterField
        points={XS.map((x, i) => ({ x, y: residuals[i]!, id: i, color: TS[i] === 1 ? tokens.series[1] : tokens.series[0] }))}
        size={4.5}
      />
    </Plot>
  );
}
