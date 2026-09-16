import { fileURLToPath } from 'node:url';
import path from 'node:path';

export const GATES_DIR = fileURLToPath(new URL('..', import.meta.url));
export const ROOT = path.resolve(GATES_DIR, '..', '..');

export const CONTENT_DIR = path.join(ROOT, 'content');
export const SECTIONS_DIR = path.join(CONTENT_DIR, 'sections');
export const CONCEPTS_YAML = path.join(CONTENT_DIR, 'concepts.yaml');
export const CHAPTERS_YAML = path.join(CONTENT_DIR, 'chapters.yaml');

export const EQUATIONS_MANIFEST = path.join(ROOT, 'tools', 'extract', 'equations.json');

export const WEB_DIR = path.join(ROOT, 'apps', 'web');
export const CONTENT_CONFIG = path.join(WEB_DIR, 'src', 'content.config.ts');
export const DIST_DIR = path.join(WEB_DIR, 'dist');

export const ROOTS_JSON = path.join(GATES_DIR, 'roots.json');
export const BUDGETS_JSON = path.join(GATES_DIR, 'budgets.json');
export const FIXTURES_DIR = path.join(GATES_DIR, '__fixtures__');
