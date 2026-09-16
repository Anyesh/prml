import { useEffect, useMemo, useState } from 'react';
import { crossEntropyError, irlsFit, norm, pcg32, standardNormal } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider, StepThrough } from '@prml/ui';
import '../widgets.css';

const N_PER_CLASS = 25;
const LINE_X = [-6, 6];
const MAX_ITERATIONS = 12;

function buildDataset(separation: number) {
  const rng = pcg32(20260430);
  const class0 = Array.from({ length: N_PER_CLASS }, () => [
    1,
    -separation / 2 + standardNormal(rng),
    standardNormal(rng),
  ]);
  const class1 = Array.from({ length: N_PER_CLASS }, () => [
    1,
    separation / 2 + standardNormal(rng),
    standardNormal(rng),
  ]);
  const design = [...class0, ...class1];
  const targets = [...class0.map(() => 0), ...class1.map(() => 1)];
  return { design, targets, class0, class1 };
}

function boundary(w: readonly number[]): readonly [number, number][] {
  if (Math.abs(w[2]!) < 1e-9) return [];
  return LINE_X.map((x) => [x, -(w[0]! + w[1]! * x) / w[2]!] as const);
}

export default function IrlsConvergence() {
  const [separation, setSeparation] = useState(3);
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const { design, targets, class0, class1 } = useMemo(() => buildDataset(separation), [separation]);
  const fit = useMemo(() => irlsFit(design, targets, { maxIterations: MAX_ITERATIONS }), [design, targets]);

  const trajectory = useMemo(() => {
    const weightsPerIteration = [Array(design[0]!.length).fill(0), ...fit.history.map((h) => h.weights)];
    return weightsPerIteration.map((w) => ({ weights: w, error: crossEntropyError(design, targets, w), norm: norm(w) }));
  }, [fit, design, targets]);

  useEffect(() => setStep(0), [separation]);

  const current = trajectory[Math.min(step, trajectory.length - 1)]!;
  const errorCurve = trajectory.slice(0, step + 1).map((t, i) => [i, t.error] as const);
  const maxError = Math.max(...trajectory.map((t) => t.error));

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[-6, 6]} yDomain={[-6, 6]} equalAspect label="Decision boundary at the current IRLS iteration">
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid zeroLine />
        {boundary(current.weights).length > 0 && <Curve points={boundary(current.weights)} color={tokens.color.accent} width={2} />}
        <ScatterField points={class0.map((p, i) => ({ x: p[1]!, y: p[2]!, id: `0-${i}` }))} color={tokens.series[0]} size={3} />
        <ScatterField points={class1.map((p, i) => ({ x: p[1]!, y: p[2]!, id: `1-${i}` }))} color={tokens.series[1]} size={3} />
      </Plot>
      <Plot height={260} xDomain={[0, MAX_ITERATIONS]} yDomain={[0, maxError * 1.05]} label="Cross-entropy error against iteration">
        <Axes x={{ label: 'iteration' }} y={{ label: 'E(w)' }} grid />
        <Curve points={errorCurve} color={tokens.color.danger} width={2} />
        <ScatterField points={errorCurve.map(([x, y], i) => ({ x, y, id: i }))} color={tokens.color.danger} size={3} />
      </Plot>
      <StepThrough step={step} stepCount={trajectory.length} onStep={setStep} interval={400} />
      <Slider
        label="Class separation"
        value={separation}
        onChange={setSeparation}
        min={0.5}
        max={6}
        step={0.1}
        hint="Push this high enough and the classes become separable: ‖w‖ keeps growing instead of settling, because no finite w drives the error any lower."
      />
      <p className="widget-readout">
        {`Iteration ${step} of ${trajectory.length - 1}. ‖w‖ = ${current.norm.toFixed(2)}, error = ${current.error.toFixed(3)}.${
          separation > 4.5 ? ' At this separation the classes barely overlap and ‖w‖ has not levelled off.' : ''
        }`}
      </p>
    </div>
  );
}
