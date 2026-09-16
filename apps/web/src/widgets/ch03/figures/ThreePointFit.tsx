import { useMemo, useState } from 'react';
import { designMatrix, maximumLikelihoodWeights, polynomialBasis } from '@prml/math';
import { Annotation, Axes, Curve, FunctionCurve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Toggle } from '@prml/ui';
import '../../widgets.css';

const XS = [0, 1, 2];
const TS = [1, 3, 2];
const LINEAR_BASIS = polynomialBasis(1);
const QUADRATIC_BASIS = polynomialBasis(2);
const DOMAIN: readonly [number, number] = [-0.4, 2.4];

function evaluate(basis: (x: number) => number[], weights: number[], x: number): number {
  return basis(x).reduce((sum, v, i) => sum + v * weights[i]!, 0);
}

export default function ThreePointFit() {
  const [quadratic, setQuadratic] = useState(false);
  const tokens = useResolvedTokens();

  const weights = useMemo(() => {
    const basis = quadratic ? QUADRATIC_BASIS : LINEAR_BASIS;
    return maximumLikelihoodWeights(designMatrix(XS, basis), TS);
  }, [quadratic]);

  const basis = quadratic ? QUADRATIC_BASIS : LINEAR_BASIS;
  const predicted = XS.map((x) => evaluate(basis, weights, x));
  const residuals = TS.map((t, i) => t - predicted[i]!);

  return (
    <div className="widget-grid">
      <Plot height={220} xDomain={DOMAIN} yDomain={[0, 4]} label="Three points fit by a line, or by a quadratic">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <FunctionCurve f={(x) => evaluate(basis, weights, x)} domain={DOMAIN} color={tokens.color.accent} width={2} />
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
        {residuals.map((r, i) => (
          <Annotation
            key={i}
            x={XS[i]!}
            y={(TS[i]! + predicted[i]!) / 2}
            text={r >= 0 ? `+${r.toFixed(2)}` : r.toFixed(2)}
            color={tokens.color.danger}
            anchor="start"
            dx={8}
            plate
          />
        ))}
        <ScatterField
          points={XS.map((x, i) => ({ x, y: TS[i]!, id: i }))}
          color={tokens.color.ink}
          size={4.5}
          label={(_, i) => `data point ${i + 1}`}
        />
      </Plot>
      <Panel columns={1} dense>
        <Toggle
          label="Add the quadratic term"
          checked={quadratic}
          onChange={setQuadratic}
          hint="Three points and three coefficients: the quadratic interpolates exactly and every labelled residual above collapses to zero."
        />
        <p className="widget-readout">
          {`Fitted weights: ${weights.map((w) => w.toFixed(3)).join(', ')}. The residual beside each point is the target minus the prediction there.`}
        </p>
      </Panel>
    </div>
  );
}
