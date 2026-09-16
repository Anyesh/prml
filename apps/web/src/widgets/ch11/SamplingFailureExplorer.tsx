import { useMemo, useState } from 'react';
import {
  effectiveSampleSize,
  gammaCauchyEnvelope,
  gammaPdf,
  cauchyPdf,
  cauchyQuantile,
  importanceLogWeights,
  normalizeImportanceWeights,
  normalLogPdf,
  normalPdf,
  normalSample,
  pcg32,
  rejectionStep,
  type RejectionAttempt,
  type Rng,
} from '@prml/math';
import { Axes, Bars, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Slider, Toggle } from '@prml/ui';
import '../widgets.css';

const SHAPE_A = 4.3;
const N_ATTEMPTS = 220;
const N_IMPORTANCE_SAMPLES = 40;
const TARGET_MEAN = 2;
const TARGET_SCALE = 1;
const PROPOSAL_MEAN = 0;

function rejectionTrace(kMultiplier: number): { attempts: RejectionAttempt[]; k: number } {
  const envelope = gammaCauchyEnvelope(SHAPE_A);
  const k = envelope.k * kMultiplier;
  const sampler = {
    sampleProposal: (rng: Rng) => cauchyQuantile(rng.next(), envelope),
    proposalPdf: (z: number) => cauchyPdf(z, envelope),
    targetPdfUnnormalized: (z: number) => gammaPdf(z, { shape: SHAPE_A, rate: 1 }),
    k,
  };
  const rng = pcg32(7, 1);
  const attempts: RejectionAttempt[] = [];
  for (let i = 0; i < N_ATTEMPTS; i++) attempts.push(rejectionStep(rng, sampler));
  return { attempts, k };
}

function importanceTrace(proposalScale: number) {
  const rng = pcg32(11, 2);
  const samples = Array.from({ length: N_IMPORTANCE_SAMPLES }, () => normalSample(rng, { mu: PROPOSAL_MEAN, sigma2: proposalScale * proposalScale }));
  const logWeights = importanceLogWeights(
    samples,
    (z) => normalLogPdf(z, { mu: TARGET_MEAN, sigma2: TARGET_SCALE * TARGET_SCALE }),
    (z) => normalLogPdf(z, { mu: PROPOSAL_MEAN, sigma2: proposalScale * proposalScale }),
  );
  const weights = normalizeImportanceWeights(logWeights);
  return { samples, weights, ess: effectiveSampleSize(weights) };
}

function curvePoints(fn: (z: number) => number, lo: number, hi: number, n = 200): [number, number][] {
  return Array.from({ length: n + 1 }, (_, i) => {
    const z = lo + ((hi - lo) * i) / n;
    return [z, fn(z)] as [number, number];
  });
}

export default function SamplingFailureExplorer() {
  const [mode, setMode] = useState<'rejection' | 'importance'>('rejection');
  const [kMultiplier, setKMultiplier] = useState(1);
  const [proposalScale, setProposalScale] = useState(1);
  const tokens = useResolvedTokens();

  const rejection = useMemo(() => rejectionTrace(kMultiplier), [kMultiplier]);
  const importance = useMemo(() => importanceTrace(proposalScale), [proposalScale]);

  if (mode === 'rejection') {
    const accepted = rejection.attempts.filter((a) => a.accepted);
    const rejected = rejection.attempts.filter((a) => !a.accepted);
    const acceptanceRate = accepted.length / rejection.attempts.length;
    return (
      <div className="widget-grid">
        <Plot width={480} height={320} xDomain={[0, 22]} yDomain={[0, 0.22]} label="Gamma target with a scaled Cauchy envelope">
          <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
          <Curve points={curvePoints((z) => gammaPdf(z, { shape: SHAPE_A, rate: 1 }), 0.01, 22)} color={tokens.series[0]!} width={2.5} />
          <Curve
            points={curvePoints((z) => rejection.k * cauchyPdf(z, gammaCauchyEnvelope(SHAPE_A)), 0.01, 22)}
            color={tokens.color.inkMuted}
            width={1.5}
            dash="dashed"
          />
          <ScatterField
            points={accepted.map((a) => ({ x: a.candidate, y: a.u, color: tokens.color.success, size: 3 }))}
          />
          <ScatterField
            points={rejected.map((a) => ({ x: a.candidate, y: a.u, color: tokens.color.danger, size: 3, opacity: 0.5 }))}
          />
          <Legend
            entries={[
              { label: 'target Gam(z|4.3,1)', color: tokens.series[0]!, mark: 'line' },
              { label: 'k · q(z) envelope', color: tokens.color.inkMuted, mark: 'dashed-line' },
              { label: 'accepted', color: tokens.color.success, mark: 'dot' },
              { label: 'rejected', color: tokens.color.danger, mark: 'dot' },
            ]}
          />
        </Plot>
        <div>
          <Toggle label="Importance sampling mode" checked={false} onChange={() => setMode('importance')} hint="Switch to see the other way approximate sampling fails silently." />
          <Slider
            label="Envelope looseness (× the tightest valid k)"
            value={kMultiplier}
            onChange={setKMultiplier}
            min={1}
            max={6}
            step={0.1}
            format={(v) => `${v.toFixed(1)}×`}
            hint="1× is the book's tangent-matched envelope. Pull it up and watch the acceptance rate fall."
          />
          <p className="widget-readout">
            {accepted.length} of {rejection.attempts.length} candidates accepted: acceptance rate {(acceptanceRate * 100).toFixed(1)}%,
            against the theoretical 1/k = {((1 / rejection.k) * 100).toFixed(1)}% (PRML 11.14).
          </p>
        </div>
      </div>
    );
  }

  const sorted = importance.samples
    .map((z, i) => ({ z, w: importance.weights[i]! }))
    .sort((a, b) => a.z - b.z);

  return (
    <div className="widget-grid">
      <Plot width={480} height={320} xDomain={[-8, 8]} yDomain={[0, 0.45]} label="Target and proposal densities for importance sampling">
        <Axes x={{ label: 'z' }} y={{ label: 'density' }} grid />
        <Curve points={curvePoints((z) => normalPdf(z, { mu: TARGET_MEAN, sigma2: TARGET_SCALE ** 2 }), -8, 8)} color={tokens.series[0]!} width={2.5} />
        <Curve
          points={curvePoints((z) => normalPdf(z, { mu: PROPOSAL_MEAN, sigma2: proposalScale ** 2 }), -8, 8)}
          color={tokens.series[1]!}
          width={2}
          dash="dashed"
        />
        <Legend
          entries={[
            { label: 'target N(2, 1)', color: tokens.series[0]!, mark: 'line' },
            { label: 'proposal N(0, σ²)', color: tokens.series[1]!, mark: 'dashed-line' },
          ]}
        />
      </Plot>
      <Plot width={480} height={200} xDomain={[-8, 8]} yDomain={[0, Math.max(0.05, Math.max(...sorted.map((s) => s.w)) * 1.2)]} label="Normalized importance weight at each sample">
        <Axes x={{ label: 'z' }} y={{ label: 'weight' }} grid />
        <Bars bars={sorted.map((s) => ({ at: s.z, value: s.w, color: tokens.series[0]! }))} thickness={0.3} />
      </Plot>
      <div>
        <Toggle label="Importance sampling mode" checked={true} onChange={() => setMode('rejection')} hint="Switch back to rejection sampling's version of the same failure." />
        <Slider
          label="Proposal scale σ"
          value={proposalScale}
          onChange={setProposalScale}
          min={0.4}
          max={4}
          step={0.05}
          hint="Shrink σ until the proposal barely reaches the target's mass, and watch one weight swallow the rest."
        />
        <p className="widget-readout">
          Effective sample size: {importance.ess.toFixed(2)} out of {N_IMPORTANCE_SAMPLES} draws. The largest single weight carries{' '}
          {(Math.max(...importance.weights) * 100).toFixed(1)}% of the total.
        </p>
      </div>
    </div>
  );
}
