import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'x1', x: 0.3, y: 1, label: 'x1' },
  { id: 'x2', x: 1.7, y: 1, label: 'x2' },
  { id: 'x3', x: 3.1, y: 1, label: 'x3' },
  { id: 'x4', x: 4.5, y: 1, label: 'x4' },
];

const EDGES = [
  { from: 'x1', to: 'x2', directed: true, label: 'w21' },
  { from: 'x2', to: 'x3', directed: true, label: 'w32' },
  { from: 'x3', to: 'x4', directed: true, label: 'w43' },
];

export default function LinearGaussianChain() {
  return (
    <Plot width={340} height={160} xDomain={[0, 4.8]} yDomain={[0, 2]} equalAspect label="Each node Gaussian with mean a linear function of its one parent">
      <NetworkDiagram nodes={NODES} edges={EDGES} directed />
    </Plot>
  );
}
