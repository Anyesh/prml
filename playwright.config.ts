import { defineConfig, devices } from '@playwright/test';

const PORT = 4173;

export default defineConfig({
  testDir: 'tests/visual',
  fullyParallel: true,
  workers: 1,
  retries: 0,
  reporter: [['list']],

  webServer: {
    command: `pnpm --filter @prml/web build && node scripts/gates/serve-dist.mjs`,
    url: `http://localhost:${PORT}`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },

  use: {
    baseURL: `http://localhost:${PORT}`,
    deviceScaleFactor: 1,
    colorScheme: 'light',
    trace: 'retain-on-failure',
  },

  expect: {
    toHaveScreenshot: {
      // pixelmatch's own suggested default: perceptual colour delta before a
      // pixel counts as "different", loose enough to absorb sub-pixel font
      // hinting differences between CI runs on the same machine/browser.
      threshold: 0.2,
      // A real layout regression moves far more than 1% of a widget's
      // pixels; this only absorbs anti-aliasing jitter at curve and glyph
      // edges (SVG paths, KaTeX) between otherwise-identical renders.
      maxDiffPixelRatio: 0.01,
      animations: 'disabled',
    },
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], deviceScaleFactor: 1 },
    },
  ],
});
