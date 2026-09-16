import { designMatrix, ensembleMean, linspace, matvec, regularisedWeights } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import { COMMITTEE_BASIS, COMMITTEE_LAMBDA, bootstrapIndices, committeeMemberRng, committeeSinusoidData } from '../data.js';
import '../../widgets.css';

const M = 20;
const TEST_X = linspace(0, 1, 50);
const TRUTH = TEST_X.map((x) => Math.sin(2 * Math.PI * x));
const { x: DATA_X, t: DATA_T } = committeeSinusoidData();
const TEST_DESIGN = designMatrix(TEST_X, COMMITTEE_BASIS);

function fitMember(member: number): number[] {
  const idx = bootstrapIndices(committeeMemberRng(member), DATA_X.length);
  const design = designMatrix(idx.map((i) => DATA_X[i]!), COMMITTEE_BASIS);
  const targets = idx.map((i) => DATA_T[i]!);
  const weights = regularisedWeights(design, targets, COMMITTEE_LAMBDA);
  return matvec(TEST_DESIGN, weights);
}

const PREDICTIONS = Array.from({ length: M }, (_, m) => fitMember(m));
const AVERAGE = ensembleMean(PREDICTIONS);

const MEMBER_SQUARED_ERROR = TEST_X.map(
  (_, j) => PREDICTIONS.reduce((sum, preds) => sum + (preds[j]! - TRUTH[j]!) ** 2, 0) / M,
);
const COMMITTEE_SQUARED_ERROR = TEST_X.map((_, j) => (AVERAGE[j]! - TRUTH[j]!) ** 2);

export default function PointwiseErrorComparison() {
  const tokens = useResolvedTokens();
  const maxError = Math.max(...MEMBER_SQUARED_ERROR, ...COMMITTEE_SQUARED_ERROR);

  return (
    <Plot height={200} xDomain={[0, 1]} yDomain={[0, maxError * 1.1]} label="Squared error at each test point: mean over 20 members against the committee's own">
      <Axes x={{ label: 'x' }} y={{ label: 'squared error' }} grid />
      <Curve points={TEST_X.map((x, j) => [x, MEMBER_SQUARED_ERROR[j]!] as const)} color={tokens.series[0]!} width={2} />
      <Curve points={TEST_X.map((x, j) => [x, COMMITTEE_SQUARED_ERROR[j]!] as const)} color={tokens.series[1]!} width={2} />
      <Legend
        entries={[
          { label: 'mean member error', color: tokens.series[0]!, mark: 'line' },
          { label: 'committee error', color: tokens.series[1]!, mark: 'line' },
        ]}
      />
    </Plot>
  );
}
