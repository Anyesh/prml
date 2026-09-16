import { useMemo, useState } from 'react';
import { classifyPath, dSeparated, undirectedPaths, type DirectedGraph } from '@prml/math';
import { NetworkDiagram, Plot } from '@prml/viz';
import { Select } from '@prml/ui';
import '../widgets.css';

interface Layout {
  readonly graph: DirectedGraph;
  readonly positions: Readonly<Record<string, { x: number; y: number }>>;
  readonly query: readonly [string, string];
  readonly domain: readonly [number, number];
}

const LAYOUTS: Record<string, Layout> = {
  tailToTail: {
    graph: { nodes: ['A', 'B', 'C'], edges: [['C', 'A'], ['C', 'B']] },
    positions: { C: { x: 2, y: 2.2 }, A: { x: 0.4, y: 0.3 }, B: { x: 3.6, y: 0.3 } },
    query: ['A', 'B'],
    domain: [0, 4],
  },
  headToTail: {
    graph: { nodes: ['A', 'B', 'C'], edges: [['A', 'C'], ['C', 'B']] },
    positions: { A: { x: 0.3, y: 1 }, C: { x: 2, y: 1 }, B: { x: 3.7, y: 1 } },
    query: ['A', 'B'],
    domain: [0, 4],
  },
  headToHead: {
    graph: { nodes: ['A', 'B', 'C'], edges: [['A', 'C'], ['B', 'C']] },
    positions: { A: { x: 0.4, y: 2.2 }, B: { x: 3.6, y: 2.2 }, C: { x: 2, y: 0.3 } },
    query: ['A', 'B'],
    domain: [0, 4],
  },
  explainingAway: {
    graph: {
      nodes: ['A', 'B', 'C', 'D', 'E'],
      edges: [['A', 'C'], ['B', 'C'], ['C', 'D'], ['D', 'E'], ['A', 'E']],
    },
    positions: {
      A: { x: 0.3, y: 2.2 },
      B: { x: 0.3, y: 0.2 },
      C: { x: 2, y: 1.2 },
      D: { x: 3.6, y: 1.2 },
      E: { x: 5.2, y: 1.2 },
    },
    query: ['A', 'B'],
    domain: [0, 5.5],
  },
};

const GRAPH_OPTIONS = [
  { value: 'tailToTail', label: 'Tail-to-tail', hint: 'A common cause: C -> A, C -> B' },
  { value: 'headToTail', label: 'Head-to-tail', hint: 'A chain: A -> C -> B' },
  { value: 'headToHead', label: 'Head-to-head', hint: 'A common effect: A -> C <- B' },
  { value: 'explainingAway', label: 'Explaining away', hint: 'A collider with a downstream node' },
];

const ROLE_LABEL: Record<string, string> = {
  'tail-to-tail': 'tail-to-tail',
  'head-to-tail': 'head-to-tail',
  'head-to-head': 'head-to-head (a collider)',
};

export default function DSeparationExplorer() {
  const [graphKey, setGraphKey] = useState('headToHead');
  const [observed, setObserved] = useState<ReadonlySet<string>>(new Set());
  const [pathIndex, setPathIndex] = useState(0);

  const layout = LAYOUTS[graphKey]!;
  const [x, y] = layout.query;

  const paths = useMemo(() => undirectedPaths(layout.graph, x, y), [layout, x, y]);
  const activePath = useMemo(() => paths[Math.min(pathIndex, paths.length - 1)] ?? [], [paths, pathIndex]);
  const classified = useMemo(
    () => classifyPath(layout.graph, activePath, [...observed]),
    [layout, activePath, observed],
  );
  const separated = dSeparated(layout.graph, x, y, [...observed]);

  function selectGraph(key: string) {
    setGraphKey(key);
    setObserved(new Set());
    setPathIndex(0);
  }

  function toggleObserved(id: string) {
    if (id === x || id === y) return;
    setObserved((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  const onActivePath = new Set(activePath);
  const nodes = layout.graph.nodes.map((id) => {
    const pos = layout.positions[id]!;
    return {
      id,
      x: pos.x,
      y: pos.y,
      label: id,
      observed: observed.has(id),
      dimmed: !onActivePath.has(id),
    };
  });

  const pathEdgeKeys = new Set(
    activePath.slice(0, -1).map((n, i) => [n, activePath[i + 1]].sort().join('-')),
  );
  const edges = layout.graph.edges.map(([from, to]) => {
    const onPath = pathEdgeKeys.has([from, to].sort().join('-'));
    return {
      from,
      to,
      directed: true,
      dimmed: !onPath,
      dash: onPath && classified.blocked,
    };
  });

  return (
    <div>
      <div className="widget-grid">
        <Select label="Canonical graph" value={graphKey} options={GRAPH_OPTIONS} onChange={selectGraph} />
        {paths.length > 1 ? (
          <Select
            label="Path to inspect"
            value={String(pathIndex)}
            options={paths.map((p, i) => ({ value: String(i), label: p.join(' – ') }))}
            onChange={(v) => setPathIndex(Number(v))}
          />
        ) : (
          <p className="widget-readout">Only one path connects {x} and {y} in this graph.</p>
        )}
      </div>
      <Plot width={480} height={280} xDomain={layout.domain} yDomain={[0, 3]} equalAspect label="Click a node to observe or unobserve it">
        <NetworkDiagram
          nodes={nodes}
          edges={edges}
          directed
          onSelect={toggleObserved}
          nodeLabel={(n) => `${n.id}${observed.has(n.id) ? ', observed' : ''}`}
        />
      </Plot>
      <p className="widget-readout">
        Path {activePath.join(' – ')}: {classified.path.length === 0
          ? 'the two query nodes are adjacent, so nothing can block it.'
          : classified.path
              .map((step) => `${step.node} is ${ROLE_LABEL[step.role]}, ${step.blocking ? 'blocking' : 'passing through'}`)
              .join('; ')}
        . Overall, {x} and {y} are {separated ? 'd-separated (independent)' : 'not d-separated (dependent)'} given{' '}
        {observed.size === 0 ? 'nothing' : [...observed].sort().join(', ')}.
      </p>
    </div>
  );
}
