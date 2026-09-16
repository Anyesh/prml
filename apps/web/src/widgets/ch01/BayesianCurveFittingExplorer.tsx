import { useMemo, useState } from 'react';
import {
  designMatrix,
  dot,
  isotropicPrior,
  linspace,
  maximumLikelihoodWeights,
  pcg32,
  polynomialBasis,
  predictive,
  sampleWeights,
  standardNormal,
  weightPosterior,
} from '@prml/math';
import { Axes, Band, Curve, Legend, Plot, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider, Toggle } from '@prml/ui';
import '../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N = 10;
const ALPHA = 2e-3;
const BETA = 1 / (NOISE_STD * NOISE_STD);
const BASIS = polynomialBasis(9);
const GRID = linspace(0, 1, 161);
const SAMPLE_COUNT = 6;

const rng = pcg32(SEED, 1);
const XS = Array.from({ length: N }, (_, i) => i / (N - 1));
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));

export default function BayesianCurveFittingExplorer() {
  const [revealed, setRevealed] = useState(0);
  const [showMl, setShowMl] = useState(false);
  const tokens = useResolvedTokens();

  const xs = XS.slice(0, revealed);
  const ts = TS.slice(0, revealed);

  const posterior = useMemo(() => {
    if (revealed === 0) return isotropicPrior(10, ALPHA);
    return weightPosterior(designMatrix(XS.slice(0, revealed), BASIS), TS.slice(0, revealed), { alpha: ALPHA, beta: BETA });
  }, [revealed]);

  const curve = useMemo(
    () =>
      GRID.map((x) => {
        const { mean, variance } = predictive(BASIS(x), posterior, BETA);
        return { x, mean, std: Math.sqrt(variance) };
      }),
    [posterior],
  );
  const bandPoints = curve.map((c) => [c.x, c.mean - c.std, c.mean + c.std] as const);
  const meanPoints = curve.map((c) => [c.x, c.mean] as const);
  const truthPoints = GRID.map((x) => [x, Math.sin(2 * Math.PI * x)] as const);

  const sampleCurves = useMemo(() => {
    const draws = sampleWeights(pcg32(SEED, revealed + 50), posterior, SAMPLE_COUNT);
    return draws.map((w) => GRID.map((x) => [x, dot(BASIS(x), w)] as const));
  }, [posterior, revealed]);

  const mlCurve = useMemo(() => {
    if (revealed < 2) return null;
    const w = maximumLikelihoodWeights(designMatrix(XS.slice(0, revealed), BASIS), TS.slice(0, revealed));
    return GRID.map((x) => [x, dot(BASIS(x), w)] as const);
  }, [revealed]);

  const midStd = curve[Math.floor(curve.length / 2)]!.std;

  return (
    <div className="widget-grid">
      <Plot height={300} xDomain={[0, 1]} yDomain={[-2.2, 2.2]} label="Bayesian predictive distribution as observations arrive">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={truthPoints} color={tokens.color.inkFaint} dash="dashed" width={1} />
        <Band points={bandPoints} color={tokens.color.accent} opacity={0.18} />
        {sampleCurves.map((pts, i) => (
          <Curve key={i} points={pts} color={tokens.series[1]!} width={1} opacity={0.45} />
        ))}
        {mlCurve && showMl && <Curve points={mlCurve} color={tokens.color.danger} width={2} dash="dashed" />}
        <Curve points={meanPoints} color={tokens.color.accent} width={2} />
        <ScatterField points={xs.map((x, i) => ({ x, y: ts[i]!, id: i }))} color={tokens.color.ink} size={4.5} />
        <Legend
          entries={[
            { label: 'posterior mean', color: tokens.color.accent, mark: 'line' },
            { label: '±1σ predictive band', color: tokens.color.accent, mark: 'swatch' },
            { label: 'sampled weight draws', color: tokens.series[1]!, mark: 'line' },
            ...(mlCurve && showMl ? [{ label: 'ML point estimate', color: tokens.color.danger, mark: 'dashed-line' as const }] : []),
          ]}
          placement="top-right"
        />
      </Plot>

      <Panel columns={2} dense>
        <Slider
          label="Observations revealed"
          value={revealed}
          onChange={(v) => setRevealed(Math.round(v))}
          min={0}
          max={N}
          step={1}
          hint="0 draws straight from the prior over w."
        />
        <Toggle label="Overlay the maximum-likelihood curve" checked={showMl} onChange={setShowMl} />
        <p className="widget-readout">
          {`${revealed} of ${N} points seen. Predictive std at x=0.5 is ${midStd.toFixed(3)}: it starts at the prior's own spread and only ever approaches the noise floor of ${NOISE_STD}, never zero. ML gives one curve; the posterior gives all of these at once, weighted by how well each fits.`}
        </p>
      </Panel>
    </div>
  );
}
