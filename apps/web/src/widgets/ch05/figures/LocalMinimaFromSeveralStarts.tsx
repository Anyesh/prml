import { useMemo } from 'react';
import { Axes, ContourField, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { toyDescend, toyErrorAt, TOY_RANGE, toySurface } from '../toyErrorSurface';
import '../../widgets.css';

const STARTS: readonly (readonly [number, number])[] = [
  [-4, 4],
  [4, 4],
  [4, -4],
  [-4, -4],
  [0.3, -4],
];
const LEARNING_RATE = 0.08;
const MAX_STEPS = 200;
const SURFACE = toySurface(61);

export default function LocalMinimaFromSeveralStarts() {
  const tokens = useResolvedTokens();
  const runs = useMemo(
    () =>
      STARTS.map((start) => {
        const path = toyDescend(start, LEARNING_RATE, MAX_STEPS);
        const end = path[path.length - 1]!;
        return { path, end, error: toyErrorAt(end[0], end[1]) };
      }),
    [],
  );

  return (
    <Plot height={260} xDomain={[-TOY_RANGE, TOY_RANGE]} yDomain={[-TOY_RANGE, TOY_RANGE]} equalAspect label="Five starting points descending the same error surface">
      <ContourField data={SURFACE} levelCount={12} color={tokens.color.inkMuted} />
      <Axes x={{ label: 'w₁' }} y={{ label: 'w₂' }} grid zeroLine />
      {runs.map((run, i) => (
        <Curve key={i} points={run.path} color={tokens.series[i % tokens.series.length]!} width={1.5} />
      ))}
      {runs.map((run, i) => (
        <ScatterField key={i} points={[{ x: run.end[0], y: run.end[1], id: `end-${i}`, color: tokens.series[i % tokens.series.length]!, size: 5, shape: 'ring' }]} />
      ))}
    </Plot>
  );
}
