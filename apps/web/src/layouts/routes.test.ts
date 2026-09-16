import { describe, expect, it } from 'vitest';

import { joinBase } from './routes';

describe('joinBase', () => {
  it('leaves a root deployment untouched', () => {
    expect(joinBase('/', '/sections/1-1')).toBe('/sections/1-1');
  });

  it('prefixes a subpath deployment', () => {
    expect(joinBase('/prml', '/sections/1-1')).toBe('/prml/sections/1-1');
  });

  it('produces exactly one slash whether or not the base carries a trailing one', () => {
    // Astro normalises `base` differently depending on `trailingSlash`, so both spellings
    // reach here and a doubled slash would 404 only once deployed.
    expect(joinBase('/prml/', '/sections/1-1')).toBe('/prml/sections/1-1');
    expect(joinBase('/prml', '/sections/1-1')).toBe('/prml/sections/1-1');
  });

  it('keeps the root path addressable under a subpath', () => {
    expect(joinBase('/prml', '/')).toBe('/prml/');
    expect(joinBase('/', '/')).toBe('/');
  });

  it('handles a nested base', () => {
    expect(joinBase('/study/prml/', '/map')).toBe('/study/prml/map');
  });
});
