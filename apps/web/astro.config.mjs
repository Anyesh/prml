import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// The Cloudflare adapter is deliberately absent. Every page here prerenders, so the deploy
// is a directory upload to Workers static assets; the adapter arrives with the review-sync
// Worker and slots into the same wrangler config without re-platforming.
export default defineConfig({
  site: 'https://prml.internal',
  output: 'static',
  integrations: [
    mdx({
      remarkPlugins: [remarkMath],
      // `output: 'html'` keeps KaTeX out of the client bundle entirely: equations are
      // rendered to markup at build time, so a prose page ships zero JavaScript.
      rehypePlugins: [[rehypeKatex, { output: 'html', strict: 'error', trust: false }]],
    }),
    react(),
  ],
  markdown: {
    shikiConfig: { themes: { light: 'github-light', dark: 'github-dark' } },
  },
  build: {
    inlineStylesheets: 'auto',
  },
  vite: {
    ssr: {
      // The workspace packages ship TypeScript rather than built output, so Vite must
      // transform them instead of treating them as external CommonJS.
      noExternal: ['@prml/math', '@prml/viz', '@prml/ui'],
    },
  },
});
