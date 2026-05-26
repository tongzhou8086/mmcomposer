# TMA 3D — single-issue bulk per stage

> **Status:** stub — content TBD.

## Why

What bottleneck does this optimization address?  What is the cost of *not*
doing it on B200?

TODO

## How

The mechanism — what PTX instructions are involved, what data structures
in SMEM/TMEM/registers, what new synchronization?

TODO

## Code pattern

A complete kernel snippet at this rung of the ladder.  The reader (and
the agent) should be able to lift this directly into their own code.

```cuda
// TODO
```

## Pitfalls

The specific bugs we hit (or watched out for) when implementing this.

- TODO

## Expected perf delta

What lift this step gave us in the b1 → b41_w8 journey on B200 BF16.
Numbers are illustrative, not contracts — your shape may vary.

| Shape (M=N=K) | Before | After | Δ |
|---|---|---|---|
| TODO | — | — | — |

