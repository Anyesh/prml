import { linspace } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { K, mdnParamsAt } from '../mdnToyProblem';
import '../../widgets.css';

const GRID = linspace(0.05, 0.95, 60);

export default function MixingCoefficientsAcrossInput() {
  const tokens = useResolvedTokens();
  const curves = Array.from({ length: K }, (_, k) => GRID.map((t) => [t, mdnParamsAt(t).mixing[k]!] as const));

  return (
    <Plot height={220} xDomain={[0, 1]} yDomain={[0, 1]} label="Each component's mixing coefficient across the input range">
      <Axes x={{ label: 't' }} y={{ label: 'πₖ(t)' }} grid />
      {curves.map((points, k) => (
        <Curve key={k} points={points} color={tokens.series[k % tokens.series.length]!} width={2} />
      ))}
    </Plot>
  );
}
