import { useMemo, useState } from 'react';
import { normalPdf, pcg32, sliceSteppingOut, sliceShrink, sliceVerticalLogLevel } from '@prml/math';
import { Axes, Curve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider, StepThrough } from '@prml/ui';
import '../widgets.css';

function mixturePdf(z: number): number {
  return 0.5 * normalPdf(z, { mu: -2, sigma2: 0.49 }) + 0.5 * normalPdf(z, { mu: 2, sigma2: 0.49 });
}
const logTarget = (z: number) => Math.log(Math.max(mixturePdf(z), 1e-300));

interface Frame {
  readonly lo: number;
  readonly hi: number;
  readonly candidate?: number;
  readonly accepted?: boolean;
  readonly label: string;
}

function buildFrames(z0: number, w: number, rng: ReturnType<typeof pcg32>): { frames: Frame[]; u: number } {
  const logU = sliceVerticalLogLevel(rng, logTarget, z0);
  const u = Math.exp(logU);
  const initial = { lo: z0 - w / 2, hi: z0 + w / 2 };
  const stepped = sliceSteppingOut(z0, logU, w, logTarget);
  const { proposals } = sliceShrink(rng, z0, logU, stepped, logTarget);

  const frames: Frame[] = [
    { lo: initial.lo, hi: initial.hi, label: 'initial width-w window' },
    { lo: stepped.lo, hi: stepped.hi, label: 'after stepping out' },
  ];
  let lo = stepped.lo;
  let hi = stepped.hi;
  for (const p of proposals) {
    frames.push({ lo, hi, candidate: p.candidate, accepted: p.accepted, label: p.accepted ? 'accepted' : 'shrink and retry' });
    if (!p.accepted) {
      if (p.candidate < z0) lo = p.candidate;
      else hi = p.candidate;
    }
  }
  return { frames, u };
}

export default function SliceSamplingExplorer() {
  const [w, setW] = useState(1.5);
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();

  const { frames, u } = useMemo(() => buildFrames(2.6, w, pcg32(2026, 100)), [w]);
  const clampedStep = Math.min(step, frames.length - 1);
  const frame = frames[clampedStep]!;

  const curve: [number, number][] = Array.from({ length: 300 }, (_, i) => {
    const z = -6 + (12 * i) / 299;
    return [z, mixturePdf(z)];
  });

  return (
    <div className="widget-grid">
      <Plot width={460} height={260} xDomain={[-6, 6]} yDomain={[0, 0.32]} label="Slice sampling's stepping-out and shrinkage procedure on a bimodal target">
        <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
        <Curve points={curve} color={tokens.series[0]!} width={2.5} />
        <Rule y={u} color={tokens.color.accent} label="u" />
        <Rule x={frame.lo} color={tokens.color.inkMuted} />
        <Rule x={frame.hi} color={tokens.color.inkMuted} />
        {frame.candidate !== undefined ? (
          <ScatterField
            points={[{ x: frame.candidate, y: u, color: frame.accepted ? tokens.color.success : tokens.color.danger, size: 6 }]}
          />
        ) : null}
      </Plot>
      <StepThrough step={clampedStep} stepCount={frames.length} onStep={setStep} labels={frames.map((f) => f.label)} />
      <Slider label="Initial window width w" value={w} onChange={setW} min={0.3} max={4} step={0.1} hint="A narrower w needs more stepping-out; a wider one needs more shrinking. Slice sampling pays one cost or the other automatically." />
      <p className="widget-readout">{frame.label}: interval [{frame.lo.toFixed(2)}, {frame.hi.toFixed(2)}].</p>
    </div>
  );
}
