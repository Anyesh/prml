import { designMatrix, maximumLikelihoodWeights, polynomialBasis } from '@prml/math';
import { Axes, Curve, FunctionCurve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const XS = [0.05, 0.25, 0.45, 0.6, 0.8, 0.95];
const TS = [0.75, 1.1, 0.55, 0.2, 0.65, 0.35];
const BASIS = polynomialBasis(2);
const DOMAIN: readonly [number, number] = [0, 1];

const WEIGHTS = maximumLikelihoodWeights(designMatrix(XS, BASIS), TS);

function predict(x: number): number {
  return BASIS(x).reduce((s, v, i) => s + v * WEIGHTS[i]!, 0);
}

export default function ErrorFunctionGeometry() {
  const tokens = useResolvedTokens();
  const predicted = XS.map(predict);

  return (
    <div className="widget-grid">
      <Plot height={200} xDomain={DOMAIN} yDomain={[0, 1.3]} label="Six points and a fitted curve, with the vertical displacement 1.2 sums the square of">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <FunctionCurve f={predict} domain={DOMAIN} color={tokens.color.accent} width={2} />
        {XS.map((x, i) => (
          <Curve
            key={i}
            points={[
              [x, TS[i]!],
              [x, predicted[i]!],
            ]}
            color={tokens.color.danger}
            dash="dashed"
            width={1.5}
          />
        ))}
        <ScatterField points={XS.map((x, i) => ({ x, y: TS[i]!, id: i }))} color={tokens.color.ink} size={4} />
      </Plot>
      <p className="widget-readout">
        {`Each dashed segment is one {y(xn,w) - tn}. Squaring and halving their sum is exactly 1.2; the fit shown minimises that sum over w.`}
      </p>
    </div>
  );
}
