import { NetworkDiagram, Plot, useResolvedTokens } from '@prml/viz';
import '../../widgets.css';

const POSITIONS: Record<string, { x: number; y: number }> = {
  p1: { x: 0.3, y: 2.6 },
  p2: { x: 1.7, y: 2.6 },
  x: { x: 1, y: 1.6 },
  c1: { x: 0.3, y: 0.6 },
  c2: { x: 1.7, y: 0.6 },
  cp1: { x: -1.1, y: 0.6 },
  o1: { x: 3.2, y: 1.6 },
};

const EDGES: readonly [string, string][] = [
  ['p1', 'x'],
  ['p2', 'x'],
  ['x', 'c1'],
  ['x', 'c2'],
  ['cp1', 'c1'],
];

const BLANKET = new Set(['p1', 'p2', 'c1', 'c2', 'cp1']);

export default function MarkovBlanketDiagram() {
  const tokens = useResolvedTokens();
  const nodes = Object.entries(POSITIONS).map(([id, pos]) => ({
    id,
    x: pos.x,
    y: pos.y,
    label: id,
    color: id === 'x' ? tokens.color.accent : BLANKET.has(id) ? tokens.color.accent : undefined,
    dimmed: id === 'o1',
  }));
  const edges = EDGES.map(([from, to]) => ({ from, to, directed: true }));

  return (
    <Plot width={340} height={260} xDomain={[-1.6, 3.6]} yDomain={[0, 3.2]} equalAspect label="x's Markov blanket: its parents, children, and children's other parents">
      <NetworkDiagram nodes={nodes} edges={edges} directed />
    </Plot>
  );
}
