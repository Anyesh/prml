import { useMemo, useState } from 'react';
import { evalGrid, linspace, norm, pcg32, smoFitClassifier, linearKernel, rbfKernelGamma, standardNormal, svmDecisionFunction } from '@prml/math';
import { Axes, ContourField, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Select, Slider } from '@prml/ui';
import '../widgets.css';

const LINEAR_KERNEL = linearKernel();

const DOMAIN: readonly [number, number] = [-6, 6];
const GRID = linspace(-6, 6, 70);
const RBF_GAMMA = 0.3;

function seedCluster(cx: number, cy: number, n: number, seed: number) {
  const rng = pcg32(seed);
  return Array.from({ length: n }, (_, i) => ({
    x: cx + 1.1 * standardNormal(rng),
    y: cy + 1.1 * standardNormal(rng),
    id: `${seed}-${i}`,
  }));
}

export default function MaximumMarginExplorer() {
  const [classA, setClassA] = useState(() => seedCluster(-2.2, 1.2, 6, 20270701));
  const [classB, setClassB] = useState(() => seedCluster(2.0, -1.0, 6, 20270702));
  const [C, setC] = useState(2);
  const [kernelName, setKernelName] = useState<'linear' | 'rbf'>('linear');
  const tokens = useResolvedTokens();

  const points = useMemo(() => [...classA.map((p) => [p.x, p.y]), ...classB.map((p) => [p.x, p.y])], [classA, classB]);
  const labels = useMemo(() => [...classA.map(() => 1 as const), ...classB.map(() => -1 as const)], [classA, classB]);
  const kernel = useMemo(() => (kernelName === 'linear' ? LINEAR_KERNEL : rbfKernelGamma(RBF_GAMMA)), [kernelName]);

  const fit = useMemo(
    () => smoFitClassifier(points, labels, kernel, { C, tol: 1e-10, maxIterations: 5000 }),
    [points, labels, kernel, C],
  );
  const decision = useMemo(() => svmDecisionFunction(fit, points, labels, kernel), [fit, points, labels, kernel]);
  const field = useMemo(() => evalGrid(GRID, GRID, (x, y) => decision([x, y])), [decision]);

  const supportSet = new Set(fit.supportVectors);
  const svA = classA.map((p, i) => ({ ...p, isSv: supportSet.has(i) }));
  const svB = classB.map((p, i) => ({ ...p, isSv: supportSet.has(classA.length + i) }));

  const marginWidth = useMemo(() => {
    if (kernelName !== 'linear') return null;
    const w: [number, number] = [0, 0];
    fit.alpha.forEach((a, i) => {
      if (a <= 1e-8) return;
      w[0] += a * labels[i]! * points[i]![0]!;
      w[1] += a * labels[i]! * points[i]![1]!;
    });
    const wn = norm(w);
    return wn > 1e-9 ? 2 / wn : null;
  }, [fit, labels, points, kernelName]);

  return (
    <div className="widget-grid">
      <Plot height={340} xDomain={DOMAIN} yDomain={DOMAIN} equalAspect label="Two draggable classes with the SVM boundary and margin">
        <ContourField data={field} levels={[-1, 1]} color={tokens.color.inkFaint} lineWidth={1.25} />
        <ContourField data={field} levels={[0]} color={tokens.color.ink} lineWidth={2.5} />
        <Axes x={{ label: 'x₁' }} y={{ label: 'x₂' }} grid zeroLine />
        <ScatterField
          points={svA.map((p) => ({ x: p.x, y: p.y, id: p.id, color: tokens.series[0], size: p.isSv ? 5.5 : 4 }))}
          onMove={(i, x, y) => setClassA((prev) => prev.map((p, j) => (j === i ? { ...p, x, y } : p)))}
        />
        <ScatterField
          points={svB.map((p) => ({ x: p.x, y: p.y, id: p.id, color: tokens.series[1], size: p.isSv ? 5.5 : 4 }))}
          onMove={(i, x, y) => setClassB((prev) => prev.map((p, j) => (j === i ? { ...p, x, y } : p)))}
        />
        <ScatterField
          points={[...svA, ...svB].filter((p) => p.isSv).map((p) => ({ x: p.x, y: p.y, id: `ring-${p.id}`, shape: 'ring' as const, color: tokens.color.accent, size: 8 }))}
        />
      </Plot>
      <Slider label="C (soft-margin trade-off)" value={C} onChange={setC} min={0.1} max={20} step={0.1} hint="Small C tolerates margin violations; large C pushes toward the hard margin." />
      <Select
        label="Kernel"
        value={kernelName}
        onChange={(v) => setKernelName(v as 'linear' | 'rbf')}
        options={[
          { value: 'linear', label: 'Linear' },
          { value: 'rbf', label: 'RBF (γ = 0.3)' },
        ]}
      />
      <p className="widget-readout">
        {`${fit.supportVectors.length} of ${points.length} points are support vectors (ringed).`}
        {marginWidth !== null ? ` Margin width 2/‖w‖ = ${marginWidth.toFixed(2)}.` : ''}
        {' Drag a plain dot: the boundary holds still. Drag a ringed one: everything moves.'}
      </p>
    </div>
  );
}
