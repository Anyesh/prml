import { useState } from 'react';
import { NetworkDiagram, Plot, useResolvedTokens } from '@prml/viz';
import { Toggle } from '@prml/ui';
import '../widgets.css';

export const title = 'Which nodes a mean-field update can see';
export const caption =
  'Toggle x observed. The update for any one node only ever needs its Markov blanket, so watch which nodes stay dim regardless.';
export const figure = '10.5';

const NODES = [
  { id: 'pi', x: 0, y: 1.6, label: 'pi' },
  { id: 'z', x: 1.4, y: 1.6, label: 'z' },
  { id: 'x', x: 2.8, y: 1, label: 'x' },
  { id: 'mu', x: 1.4, y: 0.2, label: 'mu' },
  { id: 'lambda', x: 2.8, y: -0.4, label: 'Lambda' },
] as const;

const EDGES = [
  { from: 'pi', to: 'z' },
  { from: 'z', to: 'x' },
  { from: 'mu', to: 'x' },
  { from: 'lambda', to: 'x' },
];

export default function FactorGraphDiagram() {
  const [observed, setObserved] = useState(true);
  const tokens = useResolvedTokens();

  const nodes = NODES.map((n) => ({ ...n, observed: n.id === 'x' ? observed : false }));

  return (
    <div>
      <Plot height={220} xDomain={[-0.6, 3.6]} yDomain={[-1, 2.2]} label="Directed graphical model for the Bayesian Gaussian mixture, PRML figure 10.5">
        <NetworkDiagram nodes={nodes} edges={EDGES} directed nodeRadius={20} />
      </Plot>
      <Toggle label="x observed" checked={observed} onChange={setObserved} hint="An unshaded x is exactly as latent as z; the update rule for every node's factor does not change." />
      <p className="widget-readout" style={{ color: tokens.color.inkMuted }}>
        Updating q(mu, Lambda) uses only x and z, its Markov blanket; it never needs to know pi directly.
      </p>
    </div>
  );
}
