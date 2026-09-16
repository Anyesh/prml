import { useMemo, useState } from 'react';
import { Axes, ContourField, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { TOY_RANGE, toyDescend, toySurface } from './toyErrorSurface';
import '../widgets.css';

const START: readonly [number, number] = [-3.5, 3.2];
const MAX_STEPS = 60;
const SURFACE = toySurface(61);

export default function ErrorSurfaceDescent() {
  const [learningRate, setLearningRate] = useState(0.05);
  const tokens = useResolvedTokens();

  const path = useMemo(() => toyDescend(START, learningRate, MAX_STEPS), [learningRate]);
  const last = path[path.length - 1]!;
  const diverged = Math.abs(last[0]) > TOY_RANGE || Math.abs(last[1]) > TOY_RANGE;

  return (
    <div className="widget-grid">
      <Plot height={320} xDomain={[-TOY_RANGE, TOY_RANGE]} yDomain={[-TOY_RANGE, TOY_RANGE]} equalAspect label="Error surface over two weights with a gradient-descent path">
        <ContourField data={SURFACE} levelCount={12} color={tokens.color.inkMuted} />
        <Axes x={{ label: 'w₁ (input to hidden)' }} y={{ label: 'w₂ (hidden to output)' }} grid zeroLine />
        <Curve points={path} color={tokens.color.accent} width={2} />
        <ScatterField points={[{ x: path[0]![0], y: path[0]![1], id: 'start', color: tokens.color.ink, size: 5 }]} />
        <ScatterField points={[{ x: last[0], y: last[1], id: 'end', color: diverged ? tokens.color.danger : tokens.color.accent, size: 6, shape: 'ring' }]} />
      </Plot>
      <Slider
        label="Learning rate η"
        value={learningRate}
        onChange={setLearningRate}
        min={0.01}
        max={1.2}
        scale="log"
        hint="Small η crawls toward the minimum in tiny, reliable steps; push it high enough and the same update overshoots the valley and diverges."
      />
      <p className="widget-readout">
        {`η = ${learningRate.toFixed(3)}. ${
          diverged
            ? `The path left the plotted range after ${path.length - 1} steps: each overshoot lands somewhere with an even steeper gradient, so the next step is larger still.`
            : `After ${path.length - 1} steps the path has settled near (${last[0].toFixed(2)}, ${last[1].toFixed(2)}).`
        }`}
      </p>
    </div>
  );
}
