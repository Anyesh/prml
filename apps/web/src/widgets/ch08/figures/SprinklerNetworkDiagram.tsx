import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const NODES = [
  { id: 'C', x: 2, y: 2.7, label: 'Cloudy' },
  { id: 'S', x: 0.6, y: 1.4, label: 'Sprinkler' },
  { id: 'R', x: 3.4, y: 1.4, label: 'Rain' },
  { id: 'W', x: 2, y: 0.2, label: 'Wet grass' },
];

const EDGES = [
  { from: 'C', to: 'S' },
  { from: 'C', to: 'R' },
  { from: 'S', to: 'W' },
  { from: 'R', to: 'W' },
];

export default function SprinklerNetworkDiagram() {
  return (
    <Plot width={280} height={240} xDomain={[0, 4]} yDomain={[0, 3]} equalAspect label="The fixed graph behind the worked example: Cloudy is the only root">
      <NetworkDiagram nodes={NODES} edges={EDGES.map((e) => ({ ...e, directed: true }))} directed />
    </Plot>
  );
}
