# Content gates

Mechanical verification for the 69-section PRML rebuild. Fourteen agents write
MDX in parallel; these gates catch the failure modes that review-by-reading
misses. All of them: take no required arguments, print a human-readable
report, support `--json`, exit `0` on pass and `1` on failure, and pass
vacuously (loudly, and with exit `0`) when there is no content yet.

Run all of gates 1/2/4/5 together:

```
node scripts/gates/run-content-gates.mjs
```

or individually via the `pnpm gate:*` / `pnpm lint:content` scripts wired in
the root `package.json`.

## Gate 1 - `frontmatter.mjs`

Validates every `content/sections/**/*.mdx` frontmatter against the frozen
zod schema in `apps/web/src/content.config.ts`. The schema is **imported**,
not duplicated: `astro:content` is a virtual module that only exists inside
Astro's Vite pipeline, so a plain Node script can't `import` the config file
directly. Astro also exports the exact same `defineCollection` from the real,
non-virtual subpath `astro/content/config`. `lib/astro-content-hook.mjs` is a
Node module-resolution hook (registered via `node:module`'s `register()`)
that redirects the `astro:content` specifier to that real subpath, so
`lib/schema.mjs` can load `collections.sections.schema` unmodified. If this
import ever fails (e.g. Astro changes that internal layout), the gate prints
the real error and suggests falling back to `pnpm --filter @prml/web build`,
which also fails on schema violations.

On top of the schema, it checks two things zod cannot express:

- the file's path encodes the same chapter/section as its frontmatter -
  `content/sections/03-linear-models-for-regression/3.3-bayesian-linear-regression.mdx`
  must carry `chapter: 3, section: "3.3"`
- no two files claim the same `section` number

## Gate 2 - `mdx-structure.mjs`

Every section body must contain these six `## ` (H2) headings, in this exact
order, each exactly once. The marker is the heading text itself,
case-insensitive, trimmed:

1. `## Intuition`
2. `## The math`
3. `## Interactive`
4. `## Worked example`
5. `## Recall`
6. `## Connections`

Headings and content inside fenced code blocks (` ``` `) are ignored, so
example code containing `##` or `$` doesn't trip the scan.

Additional checks, each scoped to the text between one required heading and
the next:

- **Intuition**: no `$...$` or `$$...$$` math, and no `<MathBlock>` tag.
- **Interactive**: at least one JSX component element that isn't `<Caption>`
  or `<MathBlock>` (the "widget"), and a caption, meaning a `<Caption>` or
  `<figcaption>` element.
- **Recall**: between 3 and 6 `<Recall>` elements.
- **MathBlock <-> frontmatter**: every `<MathBlock id="x.y">` anywhere in the
  body must appear in `frontmatter.equations[].id`, and every id in
  `frontmatter.equations` must have a matching `<MathBlock>` in the body.

## Gate 4 - `equations.mjs`

Every `equations[].id` in every section's frontmatter must exist in
`tools/extract/equations.json`. Unknown ids are reported with the file that
claims them. If the manifest is missing, the gate fails immediately with:
`run node tools/extract/extract_equations.py first` - never a stack trace.

As information (not a failure), for every chapter that has at least one
section file, it lists manifest equations from that chapter that no section
references yet. That's a coverage signal, not a bug: chapters fill in over
time.

## Gate 5 - `graph.mjs`

- `prereqs` over sections must be a DAG. On a cycle it prints the actual
  cycle (`3.1 -> 3.2 -> 3.1`), not just "a cycle exists".
- Every prereq must name a section that exists.
- Every `concepts[]` entry in a section must exist in one of the `content/concepts/*.yaml`
  shards, and no two shards may define the same id.
- Every concept's `requires` must name an existing concept, and that
  relation must also be acyclic (same cycle-printing).
- Every concept's `introducedIn` / `usedIn` must name existing sections.
- No section may be orphaned. `scripts/gates/roots.json` lists the genuine
  entry points (currently just `1.1`, since chapter 1 hasn't landed - update
  it as the early curriculum is authored). Every other section must be
  reachable by following `prereqs` forward from a root.

## Gate 7 - `search-coverage.mjs`

Requires a build: fails with `run pnpm build first` if `apps/web/dist` or
`apps/web/dist/pagefind/pagefind-entry.json` is missing.

Pagefind's per-page content lives in `apps/web/dist/pagefind/fragment/*.pf_fragment`.
These are **not** the binary search index (that needs Pagefind's WASM runtime
to read); each fragment is gzip-compressed JSON with a 12-byte
`pagefind_dcd` magic prefix before the `{`, confirmed empirically by running
Pagefind against a throwaway single-page site and inspecting the output
(`zlib.gunzipSync` handles it directly, no dependency needed). Each
decompressed fragment looks like:

```json
{"url": "/...", "content": "...", "word_count": 16, "meta": {"title": "..."}, "anchors": []}
```

Route URL shape is decided by whichever agent builds the Astro pages, not by
this gate, so sections are matched to fragments **by title**
(`frontmatter.title` against `fragment.meta.title`) rather than by URL. If
two sections happen to share a title, that section is reported as a warning
("ambiguous match, skipping strict check") instead of a false failure.
Draft sections (`frontmatter.draft: true`) are skipped entirely.

A matched section fails if its fragment's `content` is empty or whitespace.

## Gate 8 - `bundle-budget.mjs`

Same build precondition and title-matching strategy as gate 7. For every
non-draft section, it finds the built HTML page by `<title>`, sums the gzip
size of every local `<script src>` it references, and checks that against
`scripts/gates/budgets.json`:

```json
{
  "proseBudgetBytes": 0,
  "defaultWidgetBudgetBytes": 153600,
  "overrides": { "3.3": 204800 }
}
```

- **Prose pages** (empty `frontmatter.widgets`) must ship exactly
  `proseBudgetBytes` (0) bytes of JS. Any script tag at all is a failure -
  that's the entire point of static MDX pages with no client-side islands.
- **Widget pages** get `overrides[section.id] ?? defaultWidgetBudgetBytes`
  gzip bytes. The default, 150 KB gzipped, is sized for one Astro island:
  the React 19 + react-dom runtime plus one `@prml/viz` chart built on a
  couple of d3 submodules, comfortably above normal usage while still
  catching the regressions that actually happen (importing all of d3 instead
  of the two submodules a chart needs, forgetting to code-split a heavy
  widget, an accidental non-tree-shakeable import). Raise it per-section in
  `overrides` with a reason, not by raising the default.

Every run prints every page's actual figure, sorted worst-first, whether or
not the gate passes - the trend matters as much as the pass/fail line.

## `roots.json` / `budgets.json`

Both are plain JSON, owned by this gate suite, meant to be edited as the
curriculum and widget catalogue grow. Neither requires touching gate code.

## Fixtures and `verify.mjs`

`scripts/gates/__fixtures__/` holds one small, deliberately broken content
tree per gate (1, 2, 4, 5) - never real content, and excluded from the
normal gate runs because they live outside `content/sections`, not because
of any special-casing in the gates themselves. `eslint.config.js` also
excludes `**/__fixtures__/**` from lint, since these are content fixtures,
not source.

```
node scripts/gates/__fixtures__/verify.mjs
```

imports each gate's `run*Gate()` function directly against its fixture
directory and asserts the expected failure messages appear. This is how the
gates were proven to work before any real content existed.

## Known environment issue: `typescript-eslint` vs TypeScript 7

The root `package.json` pins `typescript-eslint@^8.70.0` and
`typescript@^7.0.2`. `typescript-eslint` 8.70.0 hard-fails at import time on
any TypeScript major version >= 7 (see
https://github.com/typescript-eslint/typescript-eslint/issues/10940) - this
is not a config problem, it's an unconditional version guard in
`@typescript-eslint/parser`'s compiled output. `eslint.config.js` guards the
`import('typescript-eslint')` in a try/catch so the rest of lint (and the
custom rules below) still runs against JS/MJS files instead of the whole
config throwing; but until the root `package.json` either pins TypeScript
<6.1 or a `typescript-eslint` release that supports TS 7 lands, **no `.ts`/
`.tsx` file in this repo can be linted at all**, typed or not. This needs a
decision by whoever owns the root `package.json`; it's not something a
change to this directory's owned files can fix.

Neither `@eslint/js` nor `globals` are in the approved dependency list, so
when the `typescript-eslint` import fails, `eslint.config.js` falls back to a
small hand-picked set of core rules (`no-unused-vars`, `no-undef`, `eqeqeq`,
`no-var`, `prefer-const`) plus a hand-declared list of Node globals
(`process`, `console`, `URL`, ...), so JS/MJS files - this directory's own
code included - still get real base coverage instead of none at all.

## Custom lint rules (`eslint.config.js`)

- No hex/`rgb()`/`rgba()`/`hsl()`/`hsla()`/named-CSS-colour string literals
  outside `packages/ui/src/tokens.ts`.
- No raw `<svg>`/`<canvas>` JSX in `apps/web/src/widgets/**` - must come from
  `@prml/viz`.
- Widget code (`apps/web/src/widgets/**`) may only import `@prml/math`,
  `@prml/viz`, `@prml/ui`, `react`, or relative paths.
- No `Math.random` in `packages/math/**` or `apps/web/src/widgets/**` - take
  a seeded `Rng` instead, so figures and tests stay reproducible.
- No `console.log` outside `scripts/**` and `tools/**` (other `console.*`
  methods are unrestricted).

## `playwright.config.ts`

Builds `@prml/web` then serves `apps/web/dist` with a small dependency-free
static server (`scripts/gates/serve-dist.mjs`, since `apps/web`'s own
`preview` script runs the Cloudflare Workers emulator via `wrangler dev`,
which is the wrong tool for a plain static-file visual diff server). A
single deterministic `chromium` project, `deviceScaleFactor: 1`,
`colorScheme: 'light'`, one worker, no retries, and
`expect.toHaveScreenshot({ animations: 'disabled', threshold: 0.2,
maxDiffPixelRatio: 0.01 })` - `threshold` is pixelmatch's own suggested
default for perceptual colour delta per pixel, and the 1% pixel-ratio
allowance absorbs anti-aliasing jitter at SVG/KaTeX glyph edges between
otherwise-identical renders without masking a real layout regression (those
move far more than 1% of a widget's pixels).

**`tests/visual/**` is not owned by this agent** (only `scripts/gates/**`,
`eslint.config.js`, and `playwright.config.ts` are) and was not created here.
Whoever owns that suite should follow this convention: one spec per widget,
screenshotting it at three parameter settings, with the resulting PNGs
committed under `tests/visual/__screenshots__/`.
