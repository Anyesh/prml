import { useMemo, useState } from 'react';
import {
  designMatrix,
  linspace,
  maximumLikelihoodWeights,
  meanSquaredError,
  pcg32,
  polynomialBasis,
  regularisedWeights,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Plot, Rule, ScatterField, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N_TRAIN = 10;
const N_TEST = 100;
const FIT_GRID = linspace(-0.02, 1.02, 161);
const M_VALUES = Array.from({ length: 10 }, (_, m) => m);
const LN_LAMBDA_FLOOR = -30;

function sampleSet(n: number, streamId: number): { xs: number[]; ts: number[] } {
  const rng = pcg32(SEED, streamId);
  const xs = n === 1 ? [0] : Array.from({ length: n }, (_, i) => i / (n - 1));
  const ts = xs.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));
  return { xs, ts };
}

const TRAIN = sampleSet(N_TRAIN, 1);
const TEST = sampleSet(N_TEST, 2);

function fitWeights(m: number, lnLambda: number): number[] {
  const basis = polynomialBasis(m);
  const design = designMatrix(TRAIN.xs, basis);
  if (lnLambda <= LN_LAMBDA_FLOOR) return maximumLikelihoodWeights(design, TRAIN.ts);
  return regularisedWeights(design, TRAIN.ts, Math.exp(lnLambda));
}

function rmsAt(m: number, lnLambda: number): { trainRms: number; testRms: number; weights: number[] } {
  const basis = polynomialBasis(m);
  const weights = fitWeights(m, lnLambda);
  const trainRms = Math.sqrt(meanSquaredError(designMatrix(TRAIN.xs, basis), TRAIN.ts, weights));
  const testRms = Math.sqrt(meanSquaredError(designMatrix(TEST.xs, basis), TEST.ts, weights));
  return { trainRms, testRms, weights };
}

export default function PolynomialCurveFitExplorer() {
  const [m, setM] = useState(3);
  const [lnLambda, setLnLambda] = useState(LN_LAMBDA_FLOOR);
  const tokens = useResolvedTokens();

  const fit = useMemo(() => rmsAt(m, lnLambda), [m, lnLambda]);
  const basis = useMemo(() => polynomialBasis(m), [m]);
  const fitCurve = useMemo(
    () => FIT_GRID.map((x) => [x, basis(x).reduce((s, v, i) => s + v * fit.weights[i]!, 0)] as const),
    [basis, fit.weights],
  );
  const truthCurve = useMemo(() => FIT_GRID.map((x) => [x, Math.sin(2 * Math.PI * x)] as const), []);

  const rmsCurve = useMemo(() => M_VALUES.map((mv) => rmsAt(mv, lnLambda)), [lnLambda]);

  return (
    <div className="widget-grid">
      <Plot height={260} xDomain={[-0.05, 1.05]} yDomain={[-1.6, 1.6]} label="Polynomial fit against the noisy sinusoid, current M and lambda">
        <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
        <Curve points={truthCurve} color={tokens.color.inkFaint} dash="dashed" width={1.25} />
        <Curve points={fitCurve} color={tokens.color.accent} width={2} />
        <ScatterField points={TRAIN.xs.map((x, i) => ({ x, y: TRAIN.ts[i]!, id: i }))} color={tokens.color.ink} size={4} />
      </Plot>

      <Plot height={260} xDomain={[0, 9]} yDomain={[0, 1]} label="Root-mean-square error against polynomial order, at the current lambda">
        <Axes x={{ label: 'M' }} y={{ label: 'ERMS' }} grid />
        <Curve points={rmsCurve.map((r, i) => [i, r.trainRms] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={rmsCurve.map((r, i) => [i, r.testRms] as const)} color={tokens.series[1]!} width={2} />
        <Rule x={m} color={tokens.color.ink} label={`M = ${m}`} />
      </Plot>

      <Panel columns={2} dense>
        <Slider label="Polynomial order M" value={m} onChange={setM} min={0} max={9} step={1} />
        <Slider
          label="ln λ"
          value={lnLambda}
          onChange={setLnLambda}
          min={LN_LAMBDA_FLOOR}
          max={0}
          step={1}
          format={(v) => (v <= LN_LAMBDA_FLOOR ? 'no regularisation' : v.toFixed(0))}
          hint="Left end turns regularisation off entirely."
        />
        <p className="widget-readout">
          {`Train RMS = ${fit.trainRms.toFixed(4)}, test RMS = ${fit.testRms.toFixed(4)}. ${
            fit.trainRms < 0.01 && fit.testRms > 0.3
              ? 'Training error near zero while test error stays high is over-fitting, read directly off the gap between the two curves on the right.'
              : ''
          }`}
        </p>
      </Panel>
    </div>
  );
}
