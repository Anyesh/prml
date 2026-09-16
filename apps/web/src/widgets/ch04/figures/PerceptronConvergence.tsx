import { useMemo, useState } from 'react';
import { perceptronStep, zeros, type Vec } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../../widgets.css';

const POS = [
  [1, 3], [2, 4], [1.5, 5], [3, 4.5], [2.5, 3.5], [3.5, 5.5],
] as const;
const NEG = [
  [-2, -1], [-1, -2], [-3, -2.5], [-1.5, -3.5], [-2.5, -1.5], [-3.5, -4],
] as const;
const DOMAIN: readonly [number, number] = [-6, 6];
const LINE_X = [-6, 6];

interface Visit {
  readonly point: readonly [number, number];
  readonly weightsAfter: readonly number[];
  readonly updated: boolean;
}

function buildTrajectory(): Visit[] {
  const points: (readonly [number, number])[] = [...POS, ...NEG];
  const targets: (1 | -1)[] = [...POS.map(() => 1 as const), ...NEG.map(() => -1 as const)];
  const design = points.map(([x, y]) => [1, x, y]);

  let weights: Vec = zeros(3);
  const visits: Visit[] = [];
  for (let epoch = 0; epoch < 20; epoch++) {
    let misclassified = 0;
    for (let n = 0; n < design.length; n++) {
      const step = perceptronStep(weights, design[n]!, targets[n]!);
      weights = step.weights;
      if (step.updated) misclassified++;
      visits.push({ point: points[n]!, weightsAfter: weights, updated: step.updated });
    }
    if (misclassified === 0) break;
  }
  return visits;
}

const TRAJECTORY = buildTrajectory();

function boundary(w: readonly number[]): readonly [number, number][] {
  if (Math.abs(w[2]!) < 1e-9) return [];
  return LINE_X.map((x) => [x, -(w[0]! + w[1]! * x) / w[2]!] as const);
}

export default function PerceptronConvergence() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const visit = TRAJECTORY[step]!;
  const line = useMemo(() => boundary(visit.weightsAfter), [visit]);

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Perceptron boundary after each visited point">
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid zeroLine />
        {line.length > 0 && <Curve points={line} color={tokens.color.accent} width={2} />}
        <ScatterField points={POS.map((p, i) => ({ x: p[0], y: p[1], id: `p${i}` }))} color={tokens.series[0]} size={4} />
        <ScatterField points={NEG.map((p, i) => ({ x: p[0], y: p[1], id: `n${i}` }))} color={tokens.series[1]} size={4} />
        <ScatterField
          points={[{ x: visit.point[0], y: visit.point[1], id: 'current', shape: 'ring', size: 8, color: tokens.color.danger }]}
        />
      </Plot>
      <div className="widget-grid" style={{ gridTemplateColumns: '1fr' }}>
        <StepThrough step={step} stepCount={TRAJECTORY.length} onStep={setStep} interval={500} />
        <p className="widget-readout">
          {visit.updated
            ? `Point (${visit.point[0]}, ${visit.point[1]}) was misclassified: the boundary just rotated toward it.`
            : `Point (${visit.point[0]}, ${visit.point[1]}) was already correct: no update.`}
        </p>
      </div>
    </div>
  );
}
