#!/usr/bin/env node
/**
 * Gate 5: validates the two graphs that hold the curriculum together.
 *
 *  - sections.prereqs must form a DAG; every prereq must name a real section
 *  - every section.concepts entry must exist in content/concepts/
 *  - every concept.requires entry must exist and that relation must be acyclic
 *  - every concept.introducedIn / usedIn must name a real section
 *  - every non-root section must be reachable by following prereqs forward
 *    from scripts/gates/roots.json
 */
import fs from 'node:fs';
import path from 'node:path';
import { parse as parseYaml } from 'yaml';
import { parseGateArgs, reportGate } from './lib/cli.mjs';
import { findFiles } from './lib/walk.mjs';
import { readMdxFile } from './lib/mdx.mjs';
import { SECTIONS_DIR, CONCEPTS_DIR, ROOTS_JSON } from './lib/paths.mjs';

export async function runGraphGate({
  sectionsDir = SECTIONS_DIR,
  conceptsDir = CONCEPTS_DIR,
  rootsPath = ROOTS_JSON,
} = {}) {
  const failures = [];
  const warnings = [];
  const info = [];

  const files = findFiles(sectionsDir, '.mdx');
  const sectionsById = new Map();

  for (const file of files) {
    const rel = path.relative(sectionsDir, file);
    const { hasFrontmatter, frontmatter, frontmatterError } = readMdxFile(file);
    if (!hasFrontmatter || frontmatterError || !frontmatter?.section) {
      failures.push({ file: rel, message: frontmatterError ?? 'no usable frontmatter (missing "section")' });
      continue;
    }
    sectionsById.set(frontmatter.section, {
      file: rel,
      prereqs: frontmatter.prereqs ?? [],
      concepts: frontmatter.concepts ?? [],
    });
  }

  const concepts = loadConcepts(conceptsDir, info, failures);
  const roots = loadRoots(rootsPath);

  checkPrereqsExist(sectionsById, failures, warnings);
  checkPrereqCycles(sectionsById, failures);
  checkSectionConceptsExist(sectionsById, concepts, failures);
  checkConceptRequiresExist(concepts, failures);
  checkConceptRequiresCycles(concepts, failures);
  checkConceptSectionRefsExist(concepts, sectionsById, failures);
  checkOrphans(sectionsById, roots, failures, info);

  const checked = sectionsById.size + concepts.size;
  if (checked === 0) {
    return { gate: 'graph', ok: true, checked: 0, vacuous: true, failures, warnings, info };
  }

  return { gate: 'graph', ok: failures.length === 0, checked, failures, warnings, info };
}

/**
 * The graph is sharded one YAML file per chapter so parallel chapter authors never write
 * the same file. Sharding introduces a failure the single file could not have: two shards
 * defining the same id, where the last one read would silently win.
 */
function loadConcepts(conceptsDir, info, failures) {
  const map = new Map();
  if (!fs.existsSync(conceptsDir)) {
    info.push(`${path.basename(conceptsDir)}/ not found; treating the concept graph as empty.`);
    return map;
  }

  const shards = fs
    .readdirSync(conceptsDir)
    .filter((name) => name.endsWith('.yaml'))
    .sort();

  for (const shard of shards) {
    const parsed = parseYaml(fs.readFileSync(path.join(conceptsDir, shard), 'utf8'));
    for (const entry of parsed ?? []) {
      const existing = map.get(entry.id);
      if (existing) {
        failures.push({
          file: `concepts/${shard}`,
          message: `concept "${entry.id}" is already defined in concepts/${existing.shard}`,
        });
        continue;
      }
      map.set(entry.id, {
        shard,
        requires: entry.requires ?? [],
        introducedIn: entry.introducedIn,
        usedIn: entry.usedIn ?? [],
      });
    }
  }
  return map;
}

function loadRoots(rootsPath) {
  if (!fs.existsSync(rootsPath)) return new Set();
  const parsed = JSON.parse(fs.readFileSync(rootsPath, 'utf8'));
  return new Set(parsed.roots ?? []);
}

function checkPrereqsExist(sectionsById, failures, warnings) {
  // A chapter counts as covered once any of its sections is written. A prereq pointing
  // into a covered chapter must resolve, because a typo there is a real broken link; one
  // pointing into a chapter nobody has written yet is a forward reference and only warns.
  // Failing on those would push authors to drop genuine prerequisites and never restore
  // them, which costs far more than a warning that resolves itself when the chapter lands.
  const coveredChapters = new Set([...sectionsById.keys()].map((id) => id.split('.')[0]));

  for (const [id, section] of sectionsById) {
    for (const prereq of section.prereqs) {
      if (sectionsById.has(prereq)) continue;
      const target = { file: section.file, message: `prereq "${prereq}" of section "${id}" does not exist` };
      if (coveredChapters.has(prereq.split('.')[0])) {
        failures.push(target);
      } else {
        warnings.push({
          file: section.file,
          message: `prereq "${prereq}" of section "${id}" points at chapter ${prereq.split('.')[0]}, which has no sections yet`,
        });
      }
    }
  }
}

function checkPrereqCycles(sectionsById, failures) {
  const adjacency = new Map();
  for (const [id, section] of sectionsById) {
    adjacency.set(
      id,
      section.prereqs.filter((p) => sectionsById.has(p)),
    );
  }
  const cycle = findCycle(adjacency);
  if (cycle) {
    failures.push({ message: `prereq cycle: ${cycle.join(' -> ')}` });
  }
}

function checkSectionConceptsExist(sectionsById, concepts, failures) {
  for (const [id, section] of sectionsById) {
    for (const conceptId of section.concepts) {
      if (!concepts.has(conceptId)) {
        failures.push({
          file: section.file,
          message: `section "${id}" references unknown concept "${conceptId}"`,
        });
      }
    }
  }
}

function checkConceptRequiresExist(concepts, failures) {
  for (const [id, concept] of concepts) {
    for (const req of concept.requires) {
      if (!concepts.has(req)) {
        failures.push({ message: `concept "${id}" requires unknown concept "${req}"` });
      }
    }
  }
}

function checkConceptRequiresCycles(concepts, failures) {
  const adjacency = new Map();
  for (const [id, concept] of concepts) {
    adjacency.set(
      id,
      concept.requires.filter((r) => concepts.has(r)),
    );
  }
  const cycle = findCycle(adjacency);
  if (cycle) {
    failures.push({ message: `concept requires-cycle: ${cycle.join(' -> ')}` });
  }
}

function checkConceptSectionRefsExist(concepts, sectionsById, failures) {
  for (const [id, concept] of concepts) {
    if (concept.introducedIn && !sectionsById.has(concept.introducedIn)) {
      failures.push({
        message: `concept "${id}" introducedIn "${concept.introducedIn}" does not exist`,
      });
    }
    for (const usedIn of concept.usedIn) {
      if (!sectionsById.has(usedIn)) {
        failures.push({ message: `concept "${id}" usedIn "${usedIn}" does not exist` });
      }
    }
  }
}

function checkOrphans(sectionsById, roots, failures, info) {
  if (sectionsById.size === 0) return;

  for (const root of roots) {
    if (!sectionsById.has(root)) {
      info.push(`root "${root}" from roots.json does not correspond to an existing section yet`);
    }
  }

  const leadsTo = new Map();
  for (const id of sectionsById.keys()) leadsTo.set(id, []);
  for (const [id, section] of sectionsById) {
    for (const prereq of section.prereqs) {
      if (leadsTo.has(prereq)) leadsTo.get(prereq).push(id);
    }
  }

  const reachable = new Set();
  const queue = [...roots].filter((r) => sectionsById.has(r));
  for (const r of queue) reachable.add(r);
  while (queue.length > 0) {
    const current = queue.pop();
    for (const next of leadsTo.get(current) ?? []) {
      if (!reachable.has(next)) {
        reachable.add(next);
        queue.push(next);
      }
    }
  }

  for (const [id, section] of sectionsById) {
    if (!reachable.has(id) && !roots.has(id)) {
      failures.push({
        file: section.file,
        message: `section "${id}" is orphaned: not reachable from roots.json and has no prereq chain to a root`,
      });
    }
  }
}

function findCycle(adjacency) {
  const WHITE = 0;
  const GRAY = 1;
  const BLACK = 2;
  const color = new Map([...adjacency.keys()].map((k) => [k, WHITE]));
  const stack = [];

  function visit(node) {
    color.set(node, GRAY);
    stack.push(node);
    for (const next of adjacency.get(node) ?? []) {
      if (color.get(next) === GRAY) {
        const cycleStart = stack.indexOf(next);
        return [...stack.slice(cycleStart), next];
      }
      if (color.get(next) === WHITE) {
        const found = visit(next);
        if (found) return found;
      }
    }
    stack.pop();
    color.set(node, BLACK);
    return null;
  }

  for (const node of adjacency.keys()) {
    if (color.get(node) === WHITE) {
      const found = visit(node);
      if (found) return found;
    }
  }
  return null;
}

async function main() {
  const args = parseGateArgs(process.argv.slice(2));
  const result = await runGraphGate();
  process.exit(reportGate(result, { json: args.json }));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
