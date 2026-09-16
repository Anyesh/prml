import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'A', x: 0.3, y: 2.2, label: 'A' },
  { id: 'B', x: 0.3, y: 0.2, label: 'B' },
  { id: 'C', x: 2, y: 1.2, label: 'C' },
  { id: 'D', x: 3.6, y: 1.2, label: 'D' },
];

const EDGES = [
  { from: 'A', to: 'C' },
  { from: 'B', to: 'C' },
  { from: 'C', to: 'D' },
];

export default function ColliderDescendantDiagram() {
  return (
    <Plot width={300} height={200} xDomain={[0, 4]} yDomain={[0, 2.8]} equalAspect label="D is a descendant of the collider C, not the collider itself">
      <NetworkDiagram nodes={NODES} edges={EDGES.map((e) => ({ ...e, directed: true }))} directed />
    </Plot>
  );
}
