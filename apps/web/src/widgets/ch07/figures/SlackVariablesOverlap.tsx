import { useMemo } from 'react';
import { evalGrid, linspace, smoFitClassifier, linearKernel, svmDecisionFunction } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const LINEAR_KERNEL = linearKernel();

const POS: readonly (readonly [number, number])[] = [
  [1.0, 1.2], [2.1, 1.8], [1.6, -0.3], [2.8, 1.1], [0.6, -1.4], [-0.4, 0.6],
];
const NEG: readonly (readonly [number, number])[] = [
  [-0.8, -1.1], [0.3, 0.2], [1.1, -0.2], [-1.9, 0.4], [-0.6, 1.3], [0.9, 1.0],
];
const C = 1.0;
const GRID = linspace(-3.5, 3.5, 60);

export default function SlackVariablesOverlap() {
  const tokens = useResolvedTokens();

  const { classified, field } = useMemo(() => {
    const points = [...POS, ...NEG];
    const labels = [...POS.map(() => 1 as const), ...NEG.map(() => -1 as const)];
    const fit = smoFitClassifier(points, labels, LINEAR_KERNEL, { C });
    const decision = svmDecisionFunction(fit, points, labels, LINEAR_KERNEL);
    const classified = points.map((p, i) => {
      const margin = labels[i]! * decision(p);
      const slack = Math.max(0, 1 - margin);
      return { x: p[0], y: p[1], label: labels[i]!, slack, isSv: fit.supportVectors.includes(i) };
    });
    const field = evalGrid(GRID, GRID, (x, y) => decision([x, y]));
    return { classified, field };
  }, []);

  return (
    <Plot height={260} xDomain={[-3.5, 3.5]} yDomain={[-3.5, 3.5]} equalAspect label="Slack variables under a soft margin, C = 1">
      <ContourField data={field} levels={[-1, 1]} color={tokens.color.inkFaint} lineWidth={1} />
      <ContourField data={field} levels={[0]} color={tokens.color.ink} lineWidth={2} />
      <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid />
      {classified.map((p, i) => (
        <ScatterField
          key={i}
          points={[
            {
              x: p.x,
              y: p.y,
              id: i,
              color: p.label === 1 ? tokens.series[0] : tokens.series[1],
              shape: p.slack > 1 ? 'triangle' : p.isSv ? 'ring' : 'circle',
              size: p.slack > 1 ? 5.5 : p.isSv ? 5 : 3.5,
            },
          ]}
        />
      ))}
    </Plot>
  );
}
