export function sectionSlug(section: string): string {
  return section.replace('.', '-');
}

/**
 * Split from `withBase` so the join is testable without a Vite build supplying
 * `import.meta.env`. `base` may or may not carry a trailing slash depending on Astro's
 * `trailingSlash` setting, and either spelling must yield exactly one separator.
 */
export function joinBase(base: string, path: string): string {
  const trimmed = base.endsWith('/') ? base.slice(0, -1) : base;
  return `${trimmed}${path}`;
}

/**
 * Prefixes the deployment base, so the site works both at the root of an origin and under a
 * subpath. Astro rewrites its own asset URLs from `base`, but not URLs the site builds
 * itself, so every internal link must pass through here.
 */
export function withBase(path: string): string {
  return joinBase(import.meta.env.BASE_URL, path);
}

export function sectionUrl(section: string): string {
  return withBase(`/sections/${sectionSlug(section)}`);
}

export function chapterUrl(chapter: number | string): string {
  return withBase(`/chapters/${chapter}`);
}

/**
 * `MathBlock` (from `@prml/ui`) renders the equation `id` verbatim as its own DOM id, so
 * every equation anchor must use the raw id, not a slugified form, to stay in sync with it.
 */
export function equationAnchor(id: string): string {
  return id;
}

export function conceptAnchor(id: string): string {
  return `concept-${id}`;
}
