import { useMemo } from 'react';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { toyDescend, toyErrorAt } from '../toyErrorSurface';
import '../../widgets.css';

const START: readonly [number, number] = [-3.5, 3.2];
const STEPS = 40;
const RATES = [0.02, 0.08, 0.9] as const;
const Y_CEILING = 30;

export default function ErrorVsIterationComparison() {
  const tokens = useResolvedTokens();
  const curves = useMemo(
    () =>
      RATES.map((rate) => {
        const path = toyDescend(START, rate, STEPS);
        return path.map((w, i) => [i, Math.min(toyErrorAt(w[0], w[1]), Y_CEILING)] as const);
      }),
    [],
  );

  return (
    <Plot height={220} xDomain={[0, STEPS]} yDomain={[0, Y_CEILING]} label="Error against iteration for three learning rates">
      <Axes x={{ label: 'iteration' }} y={{ label: 'E(w)' }} grid />
      {curves.map((points, i) => (
        <Curve key={i} points={points} color={tokens.series[i % tokens.series.length]!} width={2} />
      ))}
    </Plot>
  );
}
