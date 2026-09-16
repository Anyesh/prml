import { useMemo, useState } from 'react';
import {
  designMatrix,
  maximumLikelihoodWeights,
  meanSquaredError,
  pcg32,
  polynomialBasis,
  standardNormal,
} from '@prml/math';
import { Axes, Curve, Legend, Plot, Rule, useResolvedTokens } from '@prml/viz';
import { Panel, Slider } from '@prml/ui';
import '../widgets.css';

const SEED = 20260101;
const NOISE_STD = 0.2;
const N = 20;
const M_VALUES = Array.from({ length: 10 }, (_, m) => m);

const rng = pcg32(SEED, 1);
const XS = Array.from({ length: N }, (_, i) => i / (N - 1));
const TS = XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(rng));

const testRng = pcg32(SEED, 2);
const TEST_XS = Array.from({ length: 200 }, (_, i) => i / 199);
const TEST_TS = TEST_XS.map((x) => Math.sin(2 * Math.PI * x) + NOISE_STD * standardNormal(testRng));

function fit(xs: number[], ts: number[], m: number): number[] {
  return maximumLikelihoodWeights(designMatrix(xs, polynomialBasis(m)), ts);
}

function trueTestRms(m: number): number {
  const w = fit(XS, TS, m);
  return Math.sqrt(meanSquaredError(designMatrix(TEST_XS, polynomialBasis(m)), TEST_TS, w));
}

function holdoutRms(m: number): number {
  const trainXs = XS.slice(0, 16);
  const trainTs = TS.slice(0, 16);
  const valXs = XS.slice(16);
  const valTs = TS.slice(16);
  const w = fit(trainXs, trainTs, m);
  return Math.sqrt(meanSquaredError(designMatrix(valXs, polynomialBasis(m)), valTs, w));
}

function cvRms(m: number, s: number): number {
  let sumSq = 0;
  let count = 0;
  for (let foldStart = 0; foldStart < s; foldStart++) {
    const trainXs: number[] = [];
    const trainTs: number[] = [];
    const heldIdx: number[] = [];
    for (let i = 0; i < N; i++) {
      if (i % s === foldStart) heldIdx.push(i);
      else {
        trainXs.push(XS[i]!);
        trainTs.push(TS[i]!);
      }
    }
    const w = fit(trainXs, trainTs, m);
    const basis = polynomialBasis(m);
    for (const i of heldIdx) {
      const pred = basis(XS[i]!).reduce((sum, v, j) => sum + v * w[j]!, 0);
      sumSq += (pred - TS[i]!) ** 2;
      count++;
    }
  }
  return Math.sqrt(sumSq / count);
}

export default function CrossValidationVsHoldout() {
  const [s, setS] = useState(5);
  const tokens = useResolvedTokens();

  const curves = useMemo(
    () =>
      M_VALUES.map((m) => ({
        m,
        cv: cvRms(m, s),
        holdout: holdoutRms(m),
        trueTest: trueTestRms(m),
      })),
    [s],
  );

  const bestCvM = curves.reduce((best, c) => (c.cv < best.cv ? c : best)).m;
  const bestHoldoutM = curves.reduce((best, c) => (c.holdout < best.holdout ? c : best)).m;
  const bestTrueM = curves.reduce((best, c) => (c.trueTest < best.trueTest ? c : best)).m;

  return (
    <div className="widget-grid">
      <Plot height={280} xDomain={[0, 9]} yDomain={[0, 1]} label="Validation RMS estimates against the true held-out RMS, by polynomial order">
        <Axes x={{ label: 'M' }} y={{ label: 'RMS estimate' }} grid />
        <Curve points={curves.map((c) => [c.m, c.trueTest] as const)} color={tokens.color.inkFaint} dash="dashed" width={2} />
        <Curve points={curves.map((c) => [c.m, c.holdout] as const)} color={tokens.series[0]!} width={2} />
        <Curve points={curves.map((c) => [c.m, c.cv] as const)} color={tokens.series[1]!} width={2} />
        <Rule x={bestTrueM} color={tokens.color.inkFaint} />
        <Legend
          entries={[
            { label: `true test RMS (100 held-out points), best M=${bestTrueM}`, color: tokens.color.inkFaint, mark: 'dashed-line' },
            { label: `single 20% hold-out, best M=${bestHoldoutM}`, color: tokens.series[0]!, mark: 'line' },
            { label: `${s}-fold CV, best M=${bestCvM}`, color: tokens.series[1]!, mark: 'line' },
          ]}
          placement="top-left"
        />
      </Plot>
      <Panel columns={1} dense>
        <Slider label="Number of folds S" value={s} onChange={(v) => setS(Math.round(v))} min={2} max={20} step={1} hint="S = 20 is leave-one-out." />
        <p className="widget-readout">
          {`With only 4 points held out, the single split is noisy and can point at the wrong M; averaging over ${s} rotations pulls the cross-validation curve closer to the true test curve, at the cost of ${s} fits instead of 1.`}
        </p>
      </Panel>
    </div>
  );
}
