import { defineCollection } from 'astro:content';
import { file, glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { parse as parseYaml } from 'yaml';

const SECTION_ID = /^\d{1,2}\.\d{1,2}$/;
const EQUATION_ID = /^\d{1,2}\.\d{1,3}$/;
const SLUG = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

const equationRef = z.object({
  id: z.string().regex(EQUATION_ID, 'equation id must look like "3.49"'),
  name: z.string().min(3),
});

const sections = defineCollection({
  loader: glob({ pattern: '**/*.mdx', base: '../../content/sections' }),
  schema: z
    .object({
      chapter: z.number().int().min(1).max(14),
      section: z.string().regex(SECTION_ID, 'section must look like "3.3"'),
      title: z.string().min(3),
      /**
       * Printed book page range, inclusive. Cited rather than quoted: PRML is free for
       * personal use but not redistributable, so pages point at the source instead of
       * reproducing it.
       */
      bookPages: z.tuple([z.number().int().positive(), z.number().int().positive()]),
      summary: z.string().min(20).max(300),
      concepts: z.array(z.string().regex(SLUG)).min(1),
      prereqs: z.array(z.string().regex(SECTION_ID)).default([]),
      /** Cross-checked against the manifest extracted from the PDF, because agents invent plausible equation numbers. */
      equations: z.array(equationRef).default([]),
      /** Component names resolved against `src/widgets/ch<nn>/`. */
      widgets: z.array(z.string().regex(/^[A-Z][A-Za-z0-9]+$/)).default([]),
      difficulty: z.number().int().min(1).max(5),
      draft: z.boolean().default(false),
    })
    .strict()
    .refine((s) => s.bookPages[0] <= s.bookPages[1], {
      message: 'bookPages must be ascending',
      path: ['bookPages'],
    })
    .refine((s) => s.section.startsWith(`${s.chapter}.`), {
      message: 'section number must begin with its chapter number',
      path: ['section'],
    })
    .refine((s) => s.equations.every((e) => e.id.startsWith(`${s.chapter}.`)), {
      message: "equations must belong to this section's chapter",
      path: ['equations'],
    })
    .refine((s) => !s.prereqs.includes(s.section), {
      message: 'a section cannot be its own prerequisite',
      path: ['prereqs'],
    }),
});

const concepts = defineCollection({
  loader: file('../../content/concepts.yaml', { parser: yamlList }),
  schema: z
    .object({
      id: z.string().regex(SLUG),
      name: z.string().min(2),
      /** Ours, not the book's. */
      definition: z.string().min(20).max(400),
      /** The one section where this concept is first properly developed. */
      introducedIn: z.string().regex(SECTION_ID),
      usedIn: z.array(z.string().regex(SECTION_ID)).default([]),
      requires: z.array(z.string().regex(SLUG)).default([]),
      aliases: z.array(z.string()).default([]),
    })
    .strict(),
});

const chapters = defineCollection({
  loader: file('../../content/chapters.yaml', { parser: yamlChapterList }),
  schema: z
    .object({
      id: z.string().regex(/^\d{1,2}$/),
      number: z.number().int().min(1).max(14),
      title: z.string().min(3),
      /** Ours, not Bishop's chapter abstract. */
      summary: z.string().min(40).max(600),
      bookPages: z.tuple([z.number().int().positive(), z.number().int().positive()]),
    })
    .strict(),
});

export const collections = { sections, concepts, chapters };

function yamlList(text: string): Record<string, unknown>[] {
  const value: unknown = parseYaml(text);
  if (!Array.isArray(value)) throw new Error('expected a top-level YAML list');
  return value as Record<string, unknown>[];
}

/** Chapters key on their number, so the loader derives `id` rather than repeating it in the file. */
function yamlChapterList(text: string): Record<string, unknown>[] {
  return yamlList(text).map((entry) => ({ ...entry, id: String(entry['number']) }));
}
