"""Golden fixtures for packages/math/src/rng.ts.

numpy's PCG64 is a different generator (64-bit output, different multiplier/shift
constants) from the 32-bit PCG32 XSH-RR variant this package implements, so it cannot
be used as an oracle here. Instead this file is a direct transcription of O'Neill's
published reference C implementation of pcg32 (pcg-random.org, "pcg_basic.c" /
the XSH-RR 2014 paper), reimplemented in Python only so the transcription can run
without a C toolchain. It is not a numpy/scipy computation; it is the reference
algorithm itself, used to pin down the exact output sequence the TypeScript port
must reproduce bit-for-bit. The {"cases": [...]} shape and "fn"/"expected" keys still
follow tools/golden/README.md, even though the source of "expected" here is a
transcribed reference implementation rather than a scipy call.
"""

import json
from pathlib import Path

MASK64 = (1 << 64) - 1
MULT = 6364136223846793005


class Pcg32:
    def __init__(self, seed: int, seq: int):
        self.state = 0
        self.inc = ((seq << 1) | 1) & MASK64
        self._step()
        self.state = (self.state + seed) & MASK64
        self._step()

    def _step(self):
        self.state = (self.state * MULT + self.inc) & MASK64

    def next_uint32(self) -> int:
        old = self.state
        self._step()
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = (old >> 59) & 0xFFFFFFFF
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF


def outputs(seed: int, seq: int, n: int) -> list[int]:
    rng = Pcg32(seed, seq)
    return [rng.next_uint32() for _ in range(n)]


FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

canonical = outputs(42, 54, 6)
fork_stream = outputs(42, 55, 6)

cases = [
    {
        "fn": "pcg32",
        "name": "canonical",
        "seed": 42,
        "stream": 54,
        "expected": canonical,
    },
    {
        "fn": "pcg32",
        "name": "fork_stream",
        "seed": 42,
        "stream": 55,
        "expected": fork_stream,
    },
]

(FIXTURES / "rng.json").write_text(json.dumps({"cases": cases}, indent=2))
print(f"wrote {FIXTURES / 'rng.json'}")
