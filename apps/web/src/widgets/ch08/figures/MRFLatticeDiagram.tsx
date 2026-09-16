import { NetworkDiagram, Plot } from '@prml/viz';
import '../../widgets.css';

const SIZE = 3;

function id(i: number, j: number): string {
  return `n${i}${j}`;
}

const nodes = Array.from({ length: SIZE }, (_, i) =>
  Array.from({ length: SIZE }, (_, j) => ({ id: id(i, j), x: j, y: SIZE - 1 - i })),
).flat();

const edges: { from: string; to: string }[] = [];
for (let i = 0; i < SIZE; i++) {
  for (let j = 0; j < SIZE; j++) {
    if (j + 1 < SIZE) edges.push({ from: id(i, j), to: id(i, j + 1) });
    if (i + 1 < SIZE) edges.push({ from: id(i, j), to: id(i + 1, j) });
  }
}

export default function MRFLatticeDiagram() {
  return (
    <Plot width={240} height={240} xDomain={[-0.5, SIZE - 0.5]} yDomain={[-0.5, SIZE - 0.5]} equalAspect label="A lattice MRF: every undirected edge is a clique of size 2">
      <NetworkDiagram nodes={nodes} edges={edges} />
    </Plot>
  );
}
