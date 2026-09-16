import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: [
      'packages/*/src/**/*.test.ts',
      'packages/*/test/**/*.test.ts',
      // Widget-level logic: the derivations a figure depends on, and the guards that keep a
      // committed artefact in step with the code that generated it.
      'apps/web/src/**/*.test.ts',
    ],
    environment: 'node',
    testTimeout: 60_000,
  },
  resolve: {
    alias: {
      '@prml/math': fileURLToPath(new URL('./packages/math/src/index.ts', import.meta.url)),
      '@prml/viz': fileURLToPath(new URL('./packages/viz/src/index.ts', import.meta.url)),
      '@prml/ui': fileURLToPath(new URL('./packages/ui/src/index.ts', import.meta.url)),
    },
  },
});
