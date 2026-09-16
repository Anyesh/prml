import { autocorrelation, gaussianRandomWalkProposal, metropolisHastingsChain, normalLogPdf, pcg32, sliceSampleChain } from '@prml/math';
import { Bars, Axes, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const N_STEPS = 8000;
const MH_STEP = 0.35;
const SLICE_WIDTH = 1.0;
const TARGETS = [
  { label: 'narrow, sd 0.30', sigma2: 0.09 },
  { label: 'wide, sd 1.50', sigma2: 2.25 },
];

function integratedTime(samples: readonly number[]): number {
  const rho = autocorrelation(samples, 50);
  return 1 + 2 * rho.slice(1).reduce((s, r) => s + Math.max(r, 0), 0);
}

export default function SliceVsMetropolisStepSize() {
  const tokens = useResolvedTokens();

  const results = TARGETS.map((t, i) => {
    const logTarget = (z: number) => normalLogPdf(z, { mu: 0, sigma2: t.sigma2 });
    const mh = metropolisHastingsChain(pcg32(2026, 130 + i), [0], N_STEPS, {
      ...gaussianRandomWalkProposal(MH_STEP),
      targetLogPdf: (z) => logTarget(z[0]!),
    });
    const slice = sliceSampleChain(pcg32(2026, 140 + i), 0, N_STEPS, logTarget, SLICE_WIDTH);
    return { ...t, mhTau: integratedTime(mh.states.map((s) => s[0]!)), sliceTau: integratedTime(slice.states) };
  });

  const bars = results.flatMap((r, i) => [
    { at: i - 0.18, value: r.mhTau, color: tokens.series[1]! },
    { at: i + 0.18, value: r.sliceTau, color: tokens.series[0]! },
  ]);

  return (
    <div className="widget-grid">
      <Plot width={380} height={240} xDomain={[-0.7, 1.7]} yDomain={[0, 65]} label="Autocorrelation time with one fixed Metropolis step size against slice sampling, on a narrow and a wide target">
        <Axes x={{ ticks: [0, 1], format: (v) => TARGETS[v]?.label ?? '' }} y={{ label: 'autocorrelation time' }} grid />
        <Bars bars={bars} thickness={0.32} />
      </Plot>
      <p className="widget-readout">
        The same Metropolis step size (0.35) gives an autocorrelation time of {results[0]!.mhTau.toFixed(1)} on the
        narrow target it was tuned for, but {results[1]!.mhTau.toFixed(1)} on the wide one. Slice sampling, started
        with the same window both times, stays near {results[0]!.sliceTau.toFixed(1)} and {results[1]!.sliceTau.toFixed(1)}
        without retuning anything.
      </p>
    </div>
  );
}
