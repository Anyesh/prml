"""Python reference port of packages/math/src/rng.ts, for chapter 11's sampling fixtures.

Not a numpy/scipy computation: like gen_rng.py's Pcg32, this is a transcription of the
already-implemented, already-tested TypeScript algorithm (PCG32 XSH-RR plus the cached
Marsaglia-polar standardNormal, randInt, and categorical), so that a Python-side
reference implementation of a chapter-11 sampler can consume *exactly* the same random
bitstream as the TypeScript one and produce a bit-identical trajectory for a fixed seed.
This is what makes "the full sampler's output for a fixed seed" an exact golden case
instead of a statistical one: the sampler algorithm itself is written fresh here from
the book's equations, independently of the TypeScript source, and only the underlying
uniform-stream generator is intentionally the same transcription (there is no scipy
distribution that reproduces our RNG's bit sequence, and pinning the sampler to a
*different* stream would defeat the purpose of a reproducible figure).
"""

import math

MASK64 = (1 << 64) - 1
MULT = 6364136223846793005


class Rng:
    """Mirrors the `Rng` interface in packages/math/src/rng.ts: nextUint32/next/fork,
    plus the free functions (standardNormal, randInt, categorical) that take an `Rng`."""

    def __init__(self, seed: int, stream: int = 1):
        self._state = 0
        self._inc = ((stream << 1) | 1) & MASK64
        self._step()
        self._state = (self._state + seed) & MASK64
        self._step()
        self._normal_cache = None

    def _step(self):
        self._state = (self._state * MULT + self._inc) & MASK64

    def next_uint32(self) -> int:
        old = self._state
        self._step()
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = (old >> 59) & 0xFFFFFFFF
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def next(self) -> float:
        return self.next_uint32() / 4294967296.0

    def fork(self, stream_id: int) -> "Rng":
        raise NotImplementedError(
            "fork is not needed by any chapter-11 reference sampler"
        )

    def standard_normal(self) -> float:
        """Marsaglia polar method, caching the second deviate on the instance exactly as
        rng.ts's `standardNormal` caches it in a WeakMap keyed by the `Rng` object."""
        if self._normal_cache is not None:
            v = self._normal_cache
            self._normal_cache = None
            return v
        while True:
            u = 2 * self.next() - 1
            v = 2 * self.next() - 1
            s = u * u + v * v
            if 0 < s < 1:
                break
        factor = math.sqrt(-2 * math.log(s) / s)
        self._normal_cache = v * factor
        return u * factor

    def rand_int(self, n: int) -> int:
        limit = (4294967296 // n) * n
        while True:
            x = self.next_uint32()
            if x < limit:
                return x % n

    def categorical(self, weights: list) -> int:
        total = sum(weights)
        r = self.next() * total
        for i, w in enumerate(weights):
            r -= w
            if r < 0:
                return i
        return len(weights) - 1
