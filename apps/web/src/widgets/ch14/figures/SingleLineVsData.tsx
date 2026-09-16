import { maximumLikelihoodWeights } from '@prml/math';
import { Axes, FunctionCurve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { expertsRegimeChangeData } from '../data.js';
import '../../widgets.css';

const { design, t } = expertsRegimeChangeData(60);
const WEIGHTS = maximumLikelihoodWeights(design, t);

export default function SingleLineVsData() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[-2, 2]} yDomain={[-3, 3]} label="One straight line fit through data that switches regime at x = 0">
      <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
      <ScatterField points={design.map((row, i) => ({ x: row[1]!, y: t[i]!, color: tokens.color.inkFaint, size: 3 }))} />
      <FunctionCurve f={(x) => WEIGHTS[0]! + WEIGHTS[1]! * x} color={tokens.color.accent} width={2.5} />
    </Plot>
  );
}
