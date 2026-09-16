import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'x1', x: 0.3, y: 2.2, label: 'x1' },
  { id: 'x2', x: 3.7, y: 2.2, label: 'x2' },
  { id: 'x3', x: 2, y: 1.1, label: 'x3' },
  { id: 'x4', x: 2, y: 0, label: 'x4' },
];

const EDGES = [
  { from: 'x1', to: 'x3', directed: true },
  { from: 'x2', to: 'x3', directed: true },
  { from: 'x3', to: 'x4', directed: true },
];

export default function GeneralFactorization() {
  return (
    <Plot width={320} height={260} xDomain={[0, 4]} yDomain={[0, 2.5]} equalAspect label="A directed graph whose joint is p(x1)p(x2)p(x3|x1,x2)p(x4|x3)">
      <NetworkDiagram nodes={NODES} edges={EDGES} directed />
    </Plot>
  );
}
