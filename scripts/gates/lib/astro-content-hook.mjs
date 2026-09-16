/**
 * Node module-resolution hook (registered via node:module `register`).
 *
 * `astro:content` is a virtual module Astro's Vite plugin injects at build/dev
 * time; it does not exist as a real file, so plain Node cannot import
 * `content.config.ts` directly. Astro also ships the same `defineCollection`
 * under the real, non-virtual subpath `astro/content/config`, so this hook
 * redirects the virtual specifier to that real one. This lets the gates
 * import the frozen zod schema instead of hand-copying it.
 */
export async function resolve(specifier, context, nextResolve) {
  if (specifier === 'astro:content') {
    return nextResolve('astro/content/config', context);
  }
  return nextResolve(specifier, context);
}
