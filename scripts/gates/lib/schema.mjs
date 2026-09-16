import { register } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';
import { CONTENT_CONFIG } from './paths.mjs';

let cachedCollections = null;
let hookRegistered = false;

/**
 * Loads the real `collections` export of `apps/web/src/content.config.ts`,
 * schema and all, via the `astro:content` -> `astro/content/config` resolution
 * hook in `astro-content-hook.mjs`. Throws a descriptive error rather than a
 * bare stack trace if the config file is missing or the import fails.
 */
export async function loadCollections() {
  if (cachedCollections) return cachedCollections;

  if (!fs.existsSync(CONTENT_CONFIG)) {
    throw new Error(
      `content config not found at ${CONTENT_CONFIG}. This gate imports the real zod ` +
        `schema from that file instead of duplicating it; it must exist for the gate to run.`,
    );
  }

  if (!hookRegistered) {
    register(new URL('./astro-content-hook.mjs', import.meta.url));
    hookRegistered = true;
  }

  try {
    const mod = await import(pathToFileURL(CONTENT_CONFIG).href);
    if (!mod.collections?.sections?.schema) {
      throw new Error('content.config.ts loaded but collections.sections.schema is missing');
    }
    cachedCollections = mod.collections;
    return cachedCollections;
  } catch (err) {
    throw new Error(
      `failed to import the zod schema from ${CONTENT_CONFIG}: ${err.message}\n` +
        `Fall back to "pnpm --filter @prml/web build" to validate content against the schema instead.`,
    );
  }
}
