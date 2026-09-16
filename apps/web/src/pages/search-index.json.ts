import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import { sectionUrl, equationAnchor, conceptAnchor } from '../layouts/routes';

export const prerender = true;

export const GET: APIRoute = async () => {
  const sections = (await getCollection('sections')).filter((s) => !s.data.draft);
  const concepts = await getCollection('concepts');

  const sectionItems = sections.map((s) => ({
    id: s.data.section,
    title: s.data.title,
    summary: s.data.summary,
    url: sectionUrl(s.data.section),
  }));

  const conceptItems = concepts.map((c) => ({
    id: c.data.id,
    name: c.data.name,
    aliases: c.data.aliases,
    url: `${sectionUrl(c.data.introducedIn)}#${conceptAnchor(c.data.id)}`,
  }));

  const equationItems = sections.flatMap((s) =>
    s.data.equations.map((eq) => ({
      id: eq.id,
      name: eq.name,
      url: `${sectionUrl(s.data.section)}#${equationAnchor(eq.id)}`,
    })),
  );

  const body = JSON.stringify({
    sections: sectionItems,
    concepts: conceptItems,
    equations: equationItems,
  });

  return new Response(body, {
    headers: { 'Content-Type': 'application/json' },
  });
};
