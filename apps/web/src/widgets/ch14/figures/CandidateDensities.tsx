import { FunctionCurve, Axes, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { normalPdf } from '@prml/math';
import { BMA_MODELS, bmaSingleModelData } from '../data.js';
import '../../widgets.css';

const SAMPLE = bmaSingleModelData(14, 0);

export default function CandidateDensities() {
  const tokens = useResolvedTokens();

  return (
    <Plot height={200} xDomain={[-5, 5]} yDomain={[0, 0.45]} label="Two candidate Gaussian models and a sample of points">
      <Axes x={{ label: 'x' }} y={false} />
      <FunctionCurve f={(x) => normalPdf(x, BMA_MODELS[0])} color={tokens.series[0]!} width={2} />
      <FunctionCurve f={(x) => normalPdf(x, BMA_MODELS[1])} color={tokens.series[1]!} width={2} />
      <ScatterField points={SAMPLE.map((x) => ({ x, y: 0.01, color: tokens.color.ink, size: 2.5 }))} />
    </Plot>
  );
}
