import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'pi', x: 0, y: 1.6, label: 'pi', dimmed: true },
  { id: 'z', x: 1.4, y: 1.6, label: 'z', dimmed: false },
  { id: 'x', x: 2.8, y: 1, label: 'x', dimmed: false, observed: true },
  { id: 'mu', x: 1.4, y: 0.2, label: 'mu', dimmed: false },
  { id: 'lambda', x: 2.8, y: -0.4, label: 'Lambda', dimmed: false },
] as const;

const EDGES = [
  { from: 'pi', to: 'z', dimmed: true },
  { from: 'z', to: 'x' },
  { from: 'mu', to: 'x' },
  { from: 'lambda', to: 'x' },
];

export default function MarkovBlanketHighlight() {
  return (
    <Plot height={200} xDomain={[-0.6, 3.6]} yDomain={[-1, 2.2]} label="The Markov blanket of z: everything an update for q(z) actually reads">
      <NetworkDiagram nodes={NODES} edges={EDGES} directed nodeRadius={20} />
    </Plot>
  );
}
