import { useMemo, useState } from 'react';
import { forwardPass, initializeWeights, linspace, pcg32, type NetworkSpec } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import '../widgets.css';

const X_GRID = linspace(-3, 3, 121);
const SEED = 20260501;

function buildSpec(hiddenUnits: number): NetworkSpec {
  return { layerSizes: [1, hiddenUnits, 1], hiddenActivation: 'tanh', outputActivation: 'linear' };
}

export default function NetworkFunctionExplorer() {
  const [hiddenUnits, setHiddenUnits] = useState(2);
  const tokens = useResolvedTokens();

  const spec = useMemo(() => buildSpec(hiddenUnits), [hiddenUnits]);
  const weights = useMemo(() => initializeWeights(spec, pcg32(SEED), 1.6), [spec]);

  const { sumCurve, unitCurves } = useMemo(() => {
    const contributions: number[][] = Array.from({ length: hiddenUnits }, () => []);
    const sum: number[] = [];
    const outRow = weights[weights.length - 1]![0]!;
    for (const x of X_GRID) {
      const trace = forwardPass(spec, weights, [x]);
      sum.push(trace.output[0]!);
      const z = trace.activations[1]!;
      for (let j = 0; j < hiddenUnits; j++) contributions[j]!.push(outRow[j + 1]! * z[j]!);
    }
    return {
      sumCurve: X_GRID.map((x, i) => [x, sum[i]!] as const),
      unitCurves: contributions.map((ys) => X_GRID.map((x, i) => [x, ys[i]!] as const)),
    };
  }, [spec, weights, hiddenUnits]);

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[-3, 3]} yDomain={[-4, 4]} label="Network function and each hidden unit's contribution">
        <Axes x={{ label: 'x' }} y={{ label: 'y(x, w)' }} grid zeroLine />
        {unitCurves.map((points, j) => (
          <Curve key={j} points={points} color={tokens.series[j % tokens.series.length]!} width={1} />
        ))}
        <Curve points={sumCurve} color={tokens.color.ink} width={2.5} />
      </Plot>
      <Slider
        label="Hidden units M"
        value={hiddenUnits}
        onChange={setHiddenUnits}
        min={1}
        max={8}
        step={1}
        hint="Each thin curve is one hidden unit's tanh bump scaled by its output weight; the bold curve is their sum plus the output bias."
      />
      <p className="widget-readout">
        {`M = ${hiddenUnits}. ${
          hiddenUnits === 1
            ? 'One tanh unit can only bend the line once.'
            : hiddenUnits <= 3
              ? 'A handful of units already buys a bump, the way one Gaussian basis function did in chapter 3, except these weights were learned rather than chosen in advance.'
              : 'More units add more independent bends: curvature keeps buying detail, at the cost of more weights to fit.'
        }`}
      </p>
    </div>
  );
}
