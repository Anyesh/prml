# Authoring a chapter

Read this end to end before writing anything, then read chapter 3 as the worked example of
every rule below.

## What you own

One chapter. Exactly these paths, and nothing else in the repo:

```
content/sections/<nn>-<chapter-slug>/<x.y>-<section-slug>.mdx
content/concepts/ch<nn>.yaml
apps/web/src/widgets/ch<nn>/**
packages/math/src/<your-area>/**                   the one directory named in your task spec
tools/golden/<yourNewModule>.py                    a fixture per new function
```

Your `@prml/math` directory is yours alone, including its `index.ts`. The root barrel
already re-exports it, so you never touch a shared barrel.

Editing a file outside that list, including shared barrels, `package.json`, design tokens,
gate scripts and `content.config.ts`, breaks other agents working at the same time. If you
need a change there, write it in your final report and stop; do not make it.

Do not run `git commit`, `git add`, or any other git command.

## Sources

| What | Where |
|---|---|
| Book text for your chapter | `tools/extract/out/ch<nn>.txt` |
| Your sections, titles, page ranges, subsections | `tools/extract/out/toc-sections.json` |
| Valid equation numbers | `tools/extract/equations.json` |
| Figure numbers and pages | `tools/extract/figures.json` |
| The exemplar | `content/sections/03-linear-models-for-regression/` |

`prml-notebooks/` is AGPL-3.0. Read it for ideas about what is worth visualising. Never copy
a line of its code; every implementation here is written fresh in TypeScript from the book's
mathematics.

PRML is free for personal use but not redistributable. Write all prose yourself and cite
page and equation numbers; never quote or paraphrase the book closely.

## Section structure

Six blocks, in this order, every section, no exceptions:

```markdown
## Intuition
## The math
## Interactive
## Worked example
## Recall
## Connections
```

- **Intuition** carries no math at all, not even inline `$\alpha$`. The gate rejects it.
  Say why the thing exists and what breaks without it. One analogy, if it earns its place.
- **The math** numbers equations to match the book, using `<MathBlock eq="4.59">`.
- **Interactive** holds the chapter's large widget in a `<Widget>` with a caption saying
  what to *do* with it, not what it shows.
- **Worked example** runs concrete numbers end to end. Every number in it must be one you
  computed, not one you estimated. See "Numbers" below.
- **Recall** holds three to six `<Recall>` cards.
- **Connections** links prereqs and successors as a short list.

## The two rules that matter most

These were asked for explicitly and are enforced by `scripts/gates/mdx-structure.mjs`.

**1. Show it, do not describe it.** Anything explicable by a graph, a chart, or a worked
example gets one. A section with a single widget dropped in to satisfy a requirement is a
failed section. The floor is three visuals; above that the requirement scales with your
prose at one visual per 130 words, and the maths block must contain at least one `<Figure>`.

**2. Cut every word that a picture already says.** Per-block word budgets:

| Block | Budget |
|---|---|
| Intuition | 170 |
| The math | 200 |
| Interactive | 70 |
| Worked example | 170 |
| Recall | 190 |
| Connections | 80 |

The two rules are coupled deliberately: adding a figure lowers what you are allowed to
write, so showing costs less than telling. Chapter 3 runs 520 to 780 prose words and four
to six visuals per section. Match that.

Small inline figures are the main vehicle. A figure that plots one relationship and nothing
else is better than a control panel with six sliders. Chapter 3 has six large widgets and
twenty-three inline figures; expect a similar ratio.

## Mounting visuals

Two wrappers, both Astro components available in every MDX file without importing them:

```mdx
import PosteriorExplorer from '@widgets/ch04/PosteriorExplorer';
import DecisionBoundaryShift from '@widgets/ch04/figures/DecisionBoundaryShift';

<Widget title="..." figure="4.12" caption="Drag a point across the boundary and watch ...">
  <PosteriorExplorer client:visible />
</Widget>

<Figure caption="..." width="wide">
  <DecisionBoundaryShift client:visible />
</Figure>
```

The `client:visible` goes on the widget, never on the wrapper. Astro can only turn the
child into an island when the wrapper owns the slot, so a React parent would render it to
dead markup with no hydration script. Widgets must be imported statically by name;
`import.meta.glob` cannot be hydrated.

`<Widget>` is chapter-scale chrome with a title and a caption. `<Figure>` is quiet: a
caption and nothing else. Use `<Figure>` far more often.

## Packages

Compose the three packages. Read the barrels before assuming something does not exist:
`packages/math/src/index.ts`, `packages/viz/src/index.ts`, `packages/ui/src/index.ts`.

- `@prml/math`: every number. Distributions, linear algebra, RNG, regression. No DOM.
- `@prml/viz`: `Plot` plus `Axes`, `Curve`, `Legend`, `ColorScale`, `ContourField`,
  `Heatmap`, `ScatterField`, `VectorField`, `ClickSurface`, `Annotation`, `Rule`, and the
  `useResolvedTokens` / `useDrag` / `usePlotClick` hooks.
- `@prml/ui`: `Slider`, `Toggle`, `StepThrough`, `Select`, `Panel`, `WidgetFrame`,
  `Figure`, `MathBlock`, `Recall`, and the design tokens.

Missing a rendering primitive? Put the request in your final report, in the format
`packages/viz/REQUESTS.md` documents, and render what you can without it in the meantime.
Do not edit `REQUESTS.md` itself: several chapters are being written at once and concurrent
writes to it lose each other. Do not mount your own `<canvas>` or write raw `<svg>` either;
lint rejects both.

## Hard rules, all lint-enforced

- No new dependencies.
- No raw `<svg>` or `<canvas>` in a widget.
- No colour literals anywhere outside `packages/ui/src/tokens.ts`. Use tokens.
- No `Math.random`. Use the seeded PCG32 `Rng` from `@prml/math`, so figures reproduce.
- No emoji in source, and no narrative comments. Comment only non-obvious *why*.
- No em dashes or double dashes in prose.

## Numbers

Every numeric claim in prose must come from a computation you ran, not from an estimate.
This is not a formality: chapter 3 shipped with wrong eigenvalues in its worked example,
and the figure that plotted the claim is what caught it. If you assert a value, plot it or
compute it.

For any new `@prml/math` function:

1. Write `tools/golden/<name>.py`, run under `uv run --with numpy --with scipy python`,
   emitting expected values as JSON.
2. Write the failing vitest against that fixture, asserted to 1e-9.
3. Then write the TypeScript.

Hand-written expected values are rejected at review. So is the reverse order.

## Concepts

Write yours in `content/concepts/ch<nn>.yaml` only. Every `concepts:` entry in your
frontmatter must resolve to a concept defined in some shard.

If a concept you need already exists in another chapter's shard, reference it; do not
redefine it, because the gate fails on an id claimed twice. If it exists but its
`introducedIn` points at a later section than yours, reference it anyway and say so in your
final report; fixing that ordering is a merge-time job, not yours.

## Frontmatter

```yaml
chapter: 4
section: "4.3"
title: Probabilistic Discriminative Models
bookPages: [203, 212]
summary: <one sentence, 20 to 300 characters>
concepts: [logistic-regression, iterative-reweighted-least-squares]
prereqs: ["4.2", "2.4"]
equations:
  - { id: "4.91", name: logistic sigmoid gradient }
widgets: [IrlsConvergence]
difficulty: 3
```

Schema is strict: unknown keys fail. `equations[].id` is cross-checked against the manifest,
so an invented-but-plausible number is caught. A `prereqs` entry pointing at an unwritten
chapter only warns, so keep genuine prerequisites in.

## Before you report done

```
pnpm lint:content     gates 1, 2, 4, 5
pnpm lint             colour, svg, canvas, Math.random, hooks
pnpm typecheck        the three packages and the site, including .astro files
pnpm test             golden fixtures
pnpm build            schema and structure, and it must succeed
```

All five must be clean. `pnpm build` does not typecheck, so a type error ships unless
`pnpm typecheck` is run too; that is what it is there for.

Assert numerics at 1e-9. If a case genuinely cannot hold there, say which one and why in
your report rather than loosening the tolerance quietly: a tolerance of 1e-6 lets a 3e-9
drift through, and that drift is exactly what these fixtures exist to catch. Then report: what you built, any concept-ordering conflicts you
found, anything you filed in `REQUESTS.md`, and anything you could not verify numerically.
Report problems plainly rather than working around them quietly.
