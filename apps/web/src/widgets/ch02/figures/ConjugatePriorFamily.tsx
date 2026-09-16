import { betaPdf, betaPosterior, linspace, sigmoid } from '@prml/math';
import { Axes, Curve, Legend, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const ETA = linspace(-6, 6, 200);
const PRIOR = { a: 3, b: 3 };
const HEADS = 9;
const TAILS = 3;
const POSTERIOR = betaPosterior(PRIOR, HEADS, TAILS);

// Change of variables from mu-space to the natural parameter eta = logit(mu): density in
// eta is the mu-density times the Jacobian |dmu/deta| = mu(1-mu), the sigmoid's own
// derivative, since d/deta sigmoid(eta) = sigmoid(eta)(1-sigmoid(eta)).
function densityInEta(eta: number, params: { a: number; b: number }): number {
  const mu = sigmoid(eta);
  return betaPdf(mu, params) * mu * (1 - mu);
}

export default function ConjugatePriorFamily() {
  const tokens = useResolvedTokens();
  return (
    <div>
      <Plot height={220} xDomain={[-6, 6]} yDomain={[0, 1]} label="Bernoulli's conjugate prior and posterior, viewed in the natural parameter eta">
        <Axes x={{ label: 'η = logit(µ)' }} y={{ label: 'density' }} grid />
        <Curve points={ETA.map((e) => [e, densityInEta(e, PRIOR)] as const)} color={tokens.color.inkFaint} width={1.75} />
        <Curve points={ETA.map((e) => [e, densityInEta(e, POSTERIOR)] as const)} color={tokens.color.accent} width={2} />
        <Legend
          entries={[
            { label: 'prior', color: tokens.color.inkFaint, mark: 'line' },
            { label: 'posterior', color: tokens.color.accent, mark: 'line' },
          ]}
          placement="top-right"
        />
      </Plot>
      <p className="widget-readout">
        {`Beta(3,3) is 2.229's general conjugate prior specialised to the Bernoulli's η; `}
        {`${HEADS} heads and ${TAILS} tails move it by exactly the sufficient statistic Σx (2.230), same shift as 2.18.`}
      </p>
    </div>
  );
}
