import { useEffect, useMemo, useState } from 'react';
import { linspace, pcg32, rvmRegressionFit, standardNormal } from '@prml/math';
import { Axes, Bars, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../widgets.css';

const N = 11;
const RVM_GAMMA = 3;

function rbfDesignRow(x: number, centers: readonly number[]): number[] {
  return [1, ...centers.map((c) => Math.exp(-RVM_GAMMA * (x - c) ** 2))];
}

function buildDataset() {
  const rng = pcg32(20270730);
  const xs = linspace(-1, 1, N);
  const trueShape = (x: number) => 0.2 + 1.5 * Math.exp(-3 * (x - 0.2) ** 2) - 1.2 * Math.exp(-3 * (x + 0.3) ** 2);
  const ts = xs.map((x) => trueShape(x) + 0.05 * standardNormal(rng));
  return { xs, ts };
}

export default function RvmSparsityAnimator() {
  const tokens = useResolvedTokens();
  const [step, setStep] = useState(0);

  const { xs, ts, design, fit } = useMemo(() => {
    const { xs, ts } = buildDataset();
    const design = xs.map((x) => rbfDesignRow(x, xs));
    const fit = rvmRegressionFit(design, ts, new Array(design[0]!.length).fill(1), 1);
    return { xs, ts, design, fit };
  }, []);

  useEffect(() => setStep(0), [fit]);

  const frame = fit.history[Math.min(step, fit.history.length - 1)]!;
  const curveXs = linspace(-1.2, 1.2, 120);
  const curve = curveXs.map((x) => {
    const row = rbfDesignRow(x, xs);
    const y = row.reduce((s, v, i) => s + v * frame.mean[i]!, 0);
    return [x, y] as const;
  });

  const relevant = frame.alpha
    .map((a, i) => ({ i, a, w: frame.mean[i]! }))
    .filter((d) => d.i > 0 && Math.abs(d.w) > 1e-3);

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[-1.2, 1.2]} yDomain={[-2, 2]} label="RVM fit at the current re-estimation sweep">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={curve} color={tokens.color.accent} width={2} />
        <ScatterField points={xs.map((x, i) => ({ x, y: ts[i]!, id: i, color: tokens.color.inkMuted, size: 3 }))} />
        <ScatterField
          points={relevant.map((d) => ({ x: xs[d.i - 1]!, y: ts[d.i - 1]!, id: `rel-${d.i}`, shape: 'ring' as const, color: tokens.color.accent, size: 7 }))}
        />
      </Plot>
      <Plot height={160} xDomain={[0, design[0]!.length - 1]} yDomain={[0, 1]} label="log10(alpha_i), rescaled to 0-1, one bar per basis function">
        <Axes x={{ label: 'basis index (0 = bias)' }} y={{ label: 'log alpha (rescaled)' }} grid />
        <Bars
          bars={frame.alpha.map((a, i) => ({
            at: i,
            value: Math.min(1, Math.log10(a + 1e-3) / 12 + 0.5),
            color: i > 0 && Math.abs(frame.mean[i]!) > 1e-3 ? tokens.color.accent : tokens.color.inkFaint,
          }))}
          thickness={0.7}
        />
      </Plot>
      <StepThrough step={step} stepCount={fit.history.length} onStep={setStep} interval={350} />
      <p className="widget-readout">
        {`Sweep ${step} of ${fit.history.length - 1}. ${relevant.length} of ${design[0]!.length - 1} basis functions remain relevant (ringed).`}
      </p>
    </div>
  );
}
