export function sectionSlug(section: string): string {
  return section.replace('.', '-');
}

export function sectionUrl(section: string): string {
  return `/sections/${sectionSlug(section)}`;
}

export function chapterUrl(chapter: number | string): string {
  return `/chapters/${chapter}`;
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
