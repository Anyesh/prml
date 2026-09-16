import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const CYCLE_NODES = [
  { id: 'A', x: 0, y: 1 },
  { id: 'B', x: 1, y: 2 },
  { id: 'C', x: 2, y: 1 },
  { id: 'D', x: 1, y: 0 },
];
const CYCLE_EDGES = [
  { from: 'A', to: 'B' },
  { from: 'B', to: 'C' },
  { from: 'C', to: 'D' },
  { from: 'D', to: 'A' },
];

const CLUSTER_NODES = [
  { id: 'ABC', x: 0.5, y: 1, label: 'ABC', shape: 'square' as const, radius: 26 },
  { id: 'ACD', x: 2.5, y: 1, label: 'ACD', shape: 'square' as const, radius: 26 },
];
const CLUSTER_EDGES = [{ from: 'ABC', to: 'ACD', label: 'AC' }];

export default function JunctionTreeDiagram() {
  return (
    <div className="widget-grid">
      <Plot width={200} height={200} xDomain={[-0.5, 2.5]} yDomain={[-0.5, 2.5]} equalAspect label="The original four-cycle">
        <NetworkDiagram nodes={CYCLE_NODES} edges={CYCLE_EDGES} />
      </Plot>
      <Plot width={200} height={120} xDomain={[-0.5, 3.5]} yDomain={[0, 2]} equalAspect label="Two overlapping clusters sharing separator AC, now a tree">
        <NetworkDiagram nodes={CLUSTER_NODES} edges={CLUSTER_EDGES} />
      </Plot>
    </div>
  );
}
