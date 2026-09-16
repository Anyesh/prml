import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'pi', x: 0, y: 1.6, label: 'pi' },
  { id: 'z', x: 1.4, y: 1.6, label: 'z' },
  { id: 'x', x: 2.8, y: 1, label: 'x', observed: true },
  { id: 'mu', x: 1.4, y: 0.2, label: 'mu' },
  { id: 'lambda', x: 2.8, y: -0.4, label: 'Lambda' },
] as const;

const EDGES = [
  { from: 'pi', to: 'z', label: 'E[ln pi]' },
  { from: 'z', to: 'x', label: 'E[z]' },
  { from: 'mu', to: 'x', label: 'E[mu]' },
  { from: 'lambda', to: 'x', label: 'E[ln|Lambda|]' },
];

export default function MessagePassingDirection() {
  return (
    <Plot height={220} xDomain={[-0.8, 3.8]} yDomain={[-1, 2.2]} label="Every update this chapter has written by hand is one of these labelled expected sufficient statistics, passed along an edge">
      <NetworkDiagram nodes={NODES} edges={EDGES} directed nodeRadius={20} />
    </Plot>
  );
}
