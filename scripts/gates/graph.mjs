#!/usr/bin/env node
/**
 * Gate 5: validates the two graphs that hold the curriculum together.
 *
 *  - sections.prereqs must form a DAG; every prereq must name a real section
 *  - every section.concepts entry must exist in content/concepts.yaml
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
import { SECTIONS_DIR, CONCEPTS_YAML, ROOTS_JSON } from './lib/paths.mjs';

export async function runGraphGate({
  sectionsDir = SECTIONS_DIR,
  conceptsPath = CONCEPTS_YAML,
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

  const concepts = loadConcepts(conceptsPath, info);
  const roots = loadRoots(rootsPath);

  checkPrereqsExist(sectionsById, failures);
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

function loadConcepts(conceptsPath, info) {
  const map = new Map();
  if (!fs.existsSync(conceptsPath)) {
    info.push(`${path.basename(conceptsPath)} not found; treating the concept graph as empty.`);
    return map;
  }
  const parsed = parseYaml(fs.readFileSync(conceptsPath, 'utf8'));
  for (const entry of parsed ?? []) {
    map.set(entry.id, {
      requires: entry.requires ?? [],
      introducedIn: entry.introducedIn,
      usedIn: entry.usedIn ?? [],
    });
  }
  return map;
}

function loadRoots(rootsPath) {
  if (!fs.existsSync(rootsPath)) return new Set();
  const parsed = JSON.parse(fs.readFileSync(rootsPath, 'utf8'));
  return new Set(parsed.roots ?? []);
}

function checkPrereqsExist(sectionsById, failures) {
  for (const [id, section] of sectionsById) {
    for (const prereq of section.prereqs) {
      if (!sectionsById.has(prereq)) {
        failures.push({
          file: section.file,
          message: `prereq "${prereq}" of section "${id}" does not exist`,
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
