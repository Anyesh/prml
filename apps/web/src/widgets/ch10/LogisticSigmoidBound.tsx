import { useMemo, useState } from 'react';
import { linspace, logisticLambda, logisticLocalBound, sigmoid } from '@prml/math';
import { Axes, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'The local bound on the logistic sigmoid';
export const caption =
  'Drag xi. The bound touches the true sigmoid at x = +/- xi and falls below it everywhere else; the two touch points slide together as one curve.';
export const figure = '10.12';

const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(DOMAIN[0], DOMAIN[1], 120);

export default function LogisticSigmoidBound() {
  const [xi, setXi] = useState(2.5);
  const tokens = useResolvedTokens();

  const sigmoidPoints = useMemo(() => GRID.map((x) => [x, sigmoid(x)] as const), []);
  const boundPoints = useMemo(() => GRID.map((x) => [x, logisticLocalBound(x, xi)] as const), [xi]);
  const lambda = logisticLambda(xi);
  const touchPoints = [
    { x: xi, y: sigmoid(xi) },
    { x: -xi, y: sigmoid(-xi) },
  ];

  return (
    <div>
      <Plot height={260} xDomain={DOMAIN} yDomain={[0, 1]} label="Logistic sigmoid and its local variational lower bound">
        <Axes x={{ label: 'x' }} y={{ label: 'sigma(x)' }} grid zeroLine />
        <Curve points={sigmoidPoints} color={tokens.color.ink} width={2} />
        <Curve points={boundPoints} color={tokens.series[0]!} width={2} dash="dashed" />
        <ScatterField points={touchPoints.map((p) => ({ x: p.x, y: p.y, color: tokens.color.danger, size: 5 }))} />
        <Legend
          entries={[
            { label: 'sigma(x)', color: tokens.color.ink, mark: 'line' },
            { label: `bound at xi = ${xi.toFixed(2)}`, color: tokens.series[0]!, mark: 'dashed-line' },
          ]}
        />
      </Plot>
      <Slider
        label="variational parameter xi"
        value={xi}
        onChange={setXi}
        min={0.1}
        max={5.5}
        step={0.05}
        format={(v) => v.toFixed(2)}
        hint="Move xi across the curve and watch both touch points travel with it, always symmetric about x = 0."
      />
      <p className="widget-readout">
        lambda(xi) = {lambda.toFixed(4)} at this xi: the coefficient of the quadratic that makes the bound exact at
        the two marked points and strictly looser everywhere else.
      </p>
    </div>
  );
}
