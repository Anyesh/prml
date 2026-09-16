import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const POSITIONS = { A: { x: 0.3, y: 1.4 }, B: { x: 1.7, y: 1.4 }, C: { x: 1, y: 0.2 } };
const nodesFor = () => Object.entries(POSITIONS).map(([id, pos]) => ({ id, x: pos.x, y: pos.y, label: id }));

export default function MoralizationDiagram() {
  return (
    <div className="widget-grid">
      <Plot width={200} height={200} xDomain={[0, 2]} yDomain={[0, 1.8]} equalAspect label="Directed: A -> C <- B, no edge between the co-parents">
        <NetworkDiagram
          nodes={nodesFor()}
          edges={[
            { from: 'A', to: 'C', directed: true },
            { from: 'B', to: 'C', directed: true },
          ]}
          directed
        />
      </Plot>
      <Plot width={200} height={200} xDomain={[0, 2]} yDomain={[0, 1.8]} equalAspect label="Moralized: an undirected edge added between A and B before the arrows are dropped">
        <NetworkDiagram
          nodes={nodesFor()}
          edges={[
            { from: 'A', to: 'C' },
            { from: 'B', to: 'C' },
            { from: 'A', to: 'B', dash: true },
          ]}
        />
      </Plot>
    </div>
  );
}
