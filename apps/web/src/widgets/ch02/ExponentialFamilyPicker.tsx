import { useMemo, useState } from 'react';
import { linspace, normalPdf } from '@prml/math';
import { Axes, Curve, Plot, useResolvedTokens } from '@prml/viz';
import { Panel, Select, Slider } from '@prml/ui';
import '../widgets.css';

export const title = 'One canonical form, several distributions';
export const caption = 'Switch distributions and watch the same h(x) g(η) exp(ηᵀu(x)) shape absorb each one.';
export const figure = '2.4';

type FamilyName = 'bernoulli' | 'gaussian';

const X = linspace(-4, 4, 200);

export default function ExponentialFamilyPicker() {
  const [family, setFamily] = useState<FamilyName>('bernoulli');
  const [mu, setMu] = useState(0.7);
  const [sigma2, setSigma2] = useState(1);
  const tokens = useResolvedTokens();

  const canonical = useMemo(() => {
    if (family === 'bernoulli') {
      const eta = Math.log(mu / (1 - mu));
      return { eta: [eta], u: 'x', h: '1', g: `σ(-η) = ${(1 / (1 + Math.exp(eta))).toFixed(3)}` };
    }
    const eta1 = mu / sigma2;
    const eta2 = -1 / (2 * sigma2);
    return {
      eta: [eta1, eta2],
      u: '(x, x²)',
      h: '(2π)^(-1/2)',
      g: `(-2η₂)^(1/2) exp(η₁²/4η₂) = ${(Math.sqrt(-2 * eta2) * Math.exp((eta1 * eta1) / (4 * eta2))).toFixed(3)}`,
    };
  }, [family, mu, sigma2]);

  return (
    <div className="widget-grid">
      <Plot height={240} xDomain={[-4, 4]} yDomain={[0, family === 'bernoulli' ? 1 : 1]} label="The distribution the current natural parameters pick out">
        <Axes x={{ label: family === 'bernoulli' ? 'x (0 or 1)' : 'x' }} y={{ label: 'p(x)' }} grid />
        {family === 'bernoulli' ? (
          <Curve
            points={[
              [0, 1 - mu],
              [0.001, 1 - mu],
              [0.999, mu],
              [1, mu],
            ]}
            color={tokens.color.accent}
            width={2}
          />
        ) : (
          <Curve points={X.map((x) => [x, normalPdf(x, { mu, sigma2 })] as const)} color={tokens.color.accent} width={2} />
        )}
      </Plot>

      <Panel columns={2} dense>
        <Select
          label="Distribution"
          value={family}
          onChange={(v) => setFamily(v as FamilyName)}
          options={[
            { value: 'bernoulli', label: 'Bernoulli' },
            { value: 'gaussian', label: 'Gaussian' },
          ]}
        />
        {family === 'bernoulli' ? (
          <Slider label="µ" value={mu} onChange={setMu} min={0.02} max={0.98} />
        ) : (
          <>
            <Slider label="µ" value={mu} onChange={setMu} min={-3} max={3} />
            <Slider label="σ²" value={sigma2} onChange={setSigma2} min={0.1} max={4} />
          </>
        )}
        <p className="widget-readout">
          {`η = (${canonical.eta.map((v) => v.toFixed(2)).join(', ')}), u(x) = ${canonical.u}, h(x) = ${canonical.h}, g(η) = ${canonical.g}`}
        </p>
      </Panel>
    </div>
  );
}
