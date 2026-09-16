import { useMemo, useState } from 'react';
import { biasVarianceDecomposition, designMatrix, ensembleMean, linspace, matvec, regularisedWeights } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Slider } from '@prml/ui';
import { COMMITTEE_BASIS, COMMITTEE_LAMBDA, bootstrapIndices, committeeMemberRng, committeeSinusoidData } from './data.js';
import '../widgets.css';

export const title = 'A committee built from bootstrap resamples';
export const caption =
  'Drag the slider to change committee size M. The left panel sprays every member and bolds their average; the right panel plots the committee error against the members\' own average error, across every M up to the slider.';
export const figure = '14.2';

const MAX_M = 30;
const TEST_X = linspace(0, 1, 50);
const TRUTH = TEST_X.map((x) => Math.sin(2 * Math.PI * x));
const { x: DATA_X, t: DATA_T } = committeeSinusoidData();
const TEST_DESIGN = designMatrix(TEST_X, COMMITTEE_BASIS);

function fitMember(member: number): number[] {
  const rng = committeeMemberRng(member);
  const idx = bootstrapIndices(rng, DATA_X.length);
  const resampledX = idx.map((i) => DATA_X[i]!);
  const resampledT = idx.map((i) => DATA_T[i]!);
  const design = designMatrix(resampledX, COMMITTEE_BASIS);
  const weights = regularisedWeights(design, resampledT, COMMITTEE_LAMBDA);
  return matvec(TEST_DESIGN, weights);
}

const MEMBER_PREDICTIONS = Array.from({ length: MAX_M }, (_, m) => fitMember(m));

function errorsUpTo(m: number) {
  const decomposition = biasVarianceDecomposition(MEMBER_PREDICTIONS.slice(0, m), TRUTH);
  return { eav: decomposition.total, ecom: decomposition.bias2 };
}

const ERROR_CURVE = Array.from({ length: MAX_M }, (_, i) => errorsUpTo(i + 1));
const NAIVE_FLOOR = ERROR_CURVE[0]!.eav;

export default function CommitteeOfBootstraps() {
  const [m, setM] = useState(6);
  const tokens = useResolvedTokens();

  const average = useMemo(() => ensembleMean(MEMBER_PREDICTIONS.slice(0, m)), [m]);
  const { eav, ecom } = ERROR_CURVE[m - 1]!;

  const eavCurve = ERROR_CURVE.map((e, i) => [i + 1, e.eav] as const);
  const ecomCurve = ERROR_CURVE.map((e, i) => [i + 1, e.ecom] as const);
  const naiveCurve = ERROR_CURVE.map((_, i) => [i + 1, NAIVE_FLOOR / (i + 1)] as const);
  const maxError = Math.max(...ERROR_CURVE.map((e) => e.eav));

  return (
    <div>
      <div className="widget-grid">
        <Plot height={260} xDomain={[0, 1]} yDomain={[-2, 2]} label="M bootstrap fits, their average, and the truth">
          <Axes x={{ label: 'x' }} y={{ label: 't' }} grid />
          {MEMBER_PREDICTIONS.slice(0, m).map((preds, i) => (
            <Curve key={i} points={TEST_X.map((x, j) => [x, preds[j]!] as const)} color={tokens.color.inkFaint} width={1} opacity={0.5} />
          ))}
          <Curve points={TEST_X.map((x, j) => [x, TRUTH[j]!] as const)} color={tokens.color.borderStrong} width={1.5} dash="dashed" />
          <Curve points={TEST_X.map((x, j) => [x, average[j]!] as const)} color={tokens.color.accent} width={3} />
        </Plot>
        <Plot height={260} xDomain={[1, MAX_M]} yDomain={[0, maxError * 1.1]} label="Committee error against average member error, across committee sizes">
          <Axes x={{ label: 'M' }} y={{ label: 'mean squared error' }} grid />
          <Curve points={eavCurve} color={tokens.series[0]!} width={2} />
          <Curve points={ecomCurve} color={tokens.series[1]!} width={2} />
          <Curve points={naiveCurve} color={tokens.color.inkMuted} width={1.5} dash="dotted" />
        </Plot>
      </div>
      <Slider label="committee size M" value={m} onChange={setM} min={1} max={MAX_M} step={1} />
      <p className="widget-readout">
        At M = {m}: average member error {eav.toFixed(4)}, committee error {ecom.toFixed(4)}. The dotted curve is
        the naive 1/M prediction from a single member's error; the committee sits above it, because the members
        do not fail independently.
      </p>
    </div>
  );
}
