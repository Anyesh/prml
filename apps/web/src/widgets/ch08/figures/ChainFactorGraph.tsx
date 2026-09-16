import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'x1', x: 0, y: 0, shape: 'circle' as const },
  { id: 'fa', x: 1, y: 0, shape: 'square' as const },
  { id: 'x2', x: 2, y: 0, shape: 'circle' as const },
  { id: 'fb', x: 3, y: 0, shape: 'square' as const },
  { id: 'x3', x: 4, y: 0, shape: 'circle' as const },
  { id: 'fc', x: 5, y: 0, shape: 'square' as const },
  { id: 'x4', x: 6, y: 0, shape: 'circle' as const },
];

const EDGES = [
  { from: 'x1', to: 'fa' },
  { from: 'fa', to: 'x2' },
  { from: 'x2', to: 'fb' },
  { from: 'fb', to: 'x3' },
  { from: 'x3', to: 'fc' },
  { from: 'fc', to: 'x4' },
];

export default function ChainFactorGraph() {
  return (
    <Plot width={340} height={110} xDomain={[-0.5, 6.5]} yDomain={[-0.8, 0.8]} equalAspect label="A chain redrawn as a factor graph: circles are variables, squares are pairwise factors">
      <NetworkDiagram nodes={NODES} edges={EDGES} />
    </Plot>
  );
}
