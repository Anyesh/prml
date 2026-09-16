import { fileURLToPath } from 'node:url';

import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// Astro 7 defaults to the Rust `satteri` markdown processor, which does not run remark or
// rehype plugins and ignores them silently. Selecting `unified` is therefore mandatory
// rather than a preference: without it no math is rendered at all, and MDX then hits the
// raw LaTeX braces and tries to parse `\mathbf{w}` as a JSX expression.
const processor = unified({
  remarkPlugins: [remarkMath],
  // `output: 'html'` keeps KaTeX out of the client bundle entirely: equations become
  // markup at build time, so a prose page ships zero JavaScript.
  rehypePlugins: [[rehypeKatex, { output: 'html', strict: 'error', trust: false }]],
  shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
});

// The Cloudflare adapter is deliberately absent. Every page prerenders, so the deploy is a
// directory upload to Workers static assets; the adapter arrives with the review-sync
// Worker and slots into the same wrangler config without re-platforming.
export default defineConfig({
  site: 'https://prml.internal',
  output: 'static',
  integrations: [mdx(), react()],
  markdown: { processor },
  build: {
    inlineStylesheets: 'auto',
  },
  vite: {
    resolve: {
      // Sections live outside the app, in content/, so they need a stable specifier for
      // the widget they import rather than a relative path climbing out of the tree.
      alias: { '@widgets': fileURLToPath(new URL('./src/widgets', import.meta.url)) },
    },
    ssr: {
      // The workspace packages ship TypeScript rather than built output, so Vite must
      // transform them instead of treating them as external CommonJS.
      noExternal: ['@prml/math', '@prml/viz', '@prml/ui'],
    },
  },
});
