import { useMemo, useState } from 'react';
import { arsBuildEnvelope, arsEnvelopeLogHeight, arsSample, pcg32, type ArsEnvelope } from '@prml/math';
import { Axes, Curve, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { StepThrough } from '@prml/ui';
import '../../widgets.css';

const DOMAIN: readonly [number, number] = [-6, 6];
const LOG_2PI = Math.log(2 * Math.PI);
const logTarget = (z: number) => -0.5 * z * z - 0.5 * LOG_2PI;
const dLogTarget = (z: number) => -z;
const INITIAL_GRID = [-2, -0.5, 0, 1, 3];
const N_SAMPLES = 6;

function buildStages(): { envelope: ArsEnvelope; rejections: number }[] {
  const rng = pcg32(2026, 1);
  let envelope: ArsEnvelope | undefined;
  const stages: { envelope: ArsEnvelope; rejections: number }[] = [
    { envelope: arsBuildEnvelope(INITIAL_GRID.map((z) => ({ z, h: logTarget(z), hPrime: dLogTarget(z) })), DOMAIN), rejections: 0 },
  ];
  let totalRejections = 0;
  for (let i = 0; i < N_SAMPLES; i++) {
    const result = arsSample(rng, { logTarget, dLogTarget, initialGrid: INITIAL_GRID, domain: DOMAIN }, envelope);
    envelope = result.envelope;
    totalRejections += result.attempts - 1;
    stages.push({ envelope, rejections: totalRejections });
  }
  return stages;
}

function envelopeCurve(envelope: ArsEnvelope, n = 200): [number, number][] {
  const [lo, hi] = DOMAIN;
  return Array.from({ length: n + 1 }, (_, i) => {
    const z = lo + ((hi - lo) * i) / n;
    return [z, Math.exp(arsEnvelopeLogHeight(envelope, z))] as [number, number];
  });
}

function targetCurve(n = 200): [number, number][] {
  const [lo, hi] = DOMAIN;
  return Array.from({ length: n + 1 }, (_, i) => {
    const z = lo + ((hi - lo) * i) / n;
    return [z, Math.exp(logTarget(z))] as [number, number];
  });
}

export default function AdaptiveRejectionEnvelope() {
  const [step, setStep] = useState(0);
  const tokens = useResolvedTokens();
  const stages = useMemo(buildStages, []);
  const stage = stages[step]!;
  const gridPoints = stage.envelope.pieces.map((p) => p.z);

  return (
    <div className="widget-grid">
      <Plot width={420} height={260} xDomain={DOMAIN} yDomain={[0, 0.55]} label="Piecewise-exponential envelope tightening around a log-concave target">
        <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
        <Curve points={targetCurve()} color={tokens.series[0]!} width={2.5} />
        <Curve points={envelopeCurve(stage.envelope)} color={tokens.color.inkMuted} width={1.5} dash="dashed" />
        <ScatterField points={gridPoints.map((z) => ({ x: z, y: Math.exp(logTarget(z)), color: tokens.color.accent, size: 4 }))} />
      </Plot>
      <StepThrough step={step} stepCount={stages.length} onStep={setStep} labels={stages.map((_, i) => (i === 0 ? 'initial grid' : `after sample ${i}`))} />
      <p className="widget-readout">
        {stage.envelope.pieces.length} grid points, {stage.rejections} rejection{stage.rejections === 1 ? '' : 's'} so
        far. Every rejection becomes a new tangent line, so the envelope never gets worse.
      </p>
    </div>
  );
}
