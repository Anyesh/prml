import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'A', x: 0, y: 1 },
  { id: 'B', x: 1, y: 2 },
  { id: 'C', x: 2, y: 1 },
  { id: 'D', x: 1, y: 0 },
];

const EDGES = [
  { from: 'A', to: 'B' },
  { from: 'B', to: 'C' },
  { from: 'C', to: 'D' },
  { from: 'D', to: 'A' },
];

export default function LoopyGraphDiagram() {
  return (
    <Plot width={220} height={220} xDomain={[-0.5, 2.5]} yDomain={[-0.5, 2.5]} equalAspect label="A four-cycle: no node splits the graph, so sum-product's schedule never terminates cleanly">
      <NetworkDiagram nodes={NODES} edges={EDGES} />
    </Plot>
  );
}
