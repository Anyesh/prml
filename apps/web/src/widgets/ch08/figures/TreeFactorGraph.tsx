import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'x1', x: 0.3, y: 2.6, shape: 'circle' as const },
  { id: 'fa', x: 1.5, y: 2.2, shape: 'square' as const },
  { id: 'x3', x: 2.7, y: 1.6, shape: 'circle' as const },
  { id: 'fb', x: 1.5, y: 1.0, shape: 'square' as const },
  { id: 'x2', x: 0.3, y: 0.6, shape: 'circle' as const },
  { id: 'fc', x: 3.9, y: 1.6, shape: 'square' as const },
  { id: 'x4', x: 5.1, y: 1.6, shape: 'circle' as const },
];

const EDGES = [
  { from: 'x1', to: 'fa' },
  { from: 'fa', to: 'x3' },
  { from: 'x2', to: 'fb' },
  { from: 'fb', to: 'x3' },
  { from: 'x3', to: 'fc' },
  { from: 'fc', to: 'x4' },
];

export default function TreeFactorGraph() {
  return (
    <Plot width={340} height={220} xDomain={[0, 5.5]} yDomain={[0, 3]} equalAspect label="x3 joined to three factors: a tree, not a chain, but still exactly one path between any two nodes">
      <NetworkDiagram nodes={NODES} edges={EDGES} />
    </Plot>
  );
}
