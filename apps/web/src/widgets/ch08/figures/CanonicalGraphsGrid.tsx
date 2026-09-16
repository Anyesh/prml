import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const GRAPHS = [
  { title: 'tail-to-tail', edges: [['C', 'A'], ['C', 'B']] as const },
  { title: 'head-to-tail', edges: [['A', 'C'], ['C', 'B']] as const },
  { title: 'head-to-head', edges: [['A', 'C'], ['B', 'C']] as const },
];

const POSITIONS = { A: { x: 0.3, y: 0.3 }, B: { x: 1.7, y: 0.3 }, C: { x: 1, y: 1.4 } };

export default function CanonicalGraphsGrid() {
  return (
    <div className="widget-grid">
      {GRAPHS.map(({ title, edges }) => (
        <Plot key={title} width={160} height={160} xDomain={[0, 2]} yDomain={[0, 1.8]} equalAspect label={title}>
          <NetworkDiagram
            nodes={Object.entries(POSITIONS).map(([id, pos]) => ({ id, x: pos.x, y: pos.y, label: id }))}
            edges={edges.map(([from, to]) => ({ from, to, directed: true }))}
            directed
          />
        </Plot>
      ))}
    </div>
  );
}
