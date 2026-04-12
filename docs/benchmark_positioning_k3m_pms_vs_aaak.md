# Benchmark Positioning for K3M Memory Evaluation

## Purpose
This document defines how benchmark results around `AAAK official`, `PMS-Bench`, and K3M-style support-first memory systems should be interpreted.

The main claim is that these benchmarks measure different kinds of realism rather than competing versions of the same metric.

## Core Position
`AAAK official` primarily measures `retrieval realism`.

`PMS-Bench` primarily measures `perceptual-memory realism`.

These are both valid aspects of real user environments, but they reward different system behaviors.

## Retrieval Realism
`AAAK official` is closest to environments where the main failure mode is:

- the system cannot find the right session in a large haystack
- the system stores too many bytes per memory item
- the system retrieval stack breaks under broad heterogeneous conversational history

This is the right benchmark when the product requirement is:

- retrieve the relevant memory shard
- compress many sessions aggressively
- survive large-scale session search

In this sense, `AAAK official` is a realistic benchmark for memory systems that behave like retrieval engines.

## Perceptual-Memory Realism
`PMS-Bench` is closest to environments where the main failure mode is:

- the system remembers the wrong preference
- the system misses a user boundary or requirement
- the system cannot recover a causal turning point
- the system returns related content but not the user's actual remembered state

This is the right benchmark when the product requirement is:

- preserve user-specific constraints
- preserve causal and turning structure
- support cue-based recall of meaningful state rather than only related session retrieval

In this sense, `PMS-Bench` is a realistic benchmark for memory systems that behave like perceptual memory.

## Why This Matters for K3M
The philosophical role of Kakeya-style support construction is not to maximize generic semantic overlap. It is to preserve a thin support carrying many future-useful continuation directions.

In text memory, these directions correspond to structures such as:

- preferred next states
- forbidden next states
- required follow-up behavior
- causal continuation
- turning-point transitions

This is why `constraint-edge fixed` aligns more naturally with `PMS-Bench` than with `AAAK official`.

Its design objective is not only:

- find a relevant session

It is also:

- preserve the structural continuation object that future queries care about

## Empirical Reading
The current empirical picture should be read as follows:

- `AAAK official` leads when the benchmark rewards high-compression retrieval over realistic haystacks.
- `constraint-edge fixed` leads when the benchmark rewards recovery of structured user memory under compression.
- These results do not invalidate one another because they target different operational definitions of success.

## Recommended Evaluation Policy
K3M should not be evaluated with only one benchmark family.

A complete evaluation stack should include:

1. `Retrieval realism`
   Measured by `AAAK official` or similar haystack retrieval benchmarks.
2. `Perceptual-memory realism`
   Measured by `PMS-Bench` or similar structure-recovery benchmarks.
3. `Compression realism`
   Measured by raw bytes, stored bytes, compression ratio, and support ratio.
4. `Serving realism`
   Measured by query latency, decode budget, and active-support size.

## Decision Rule
If the target product is a general retrieval memory system, prioritize `AAAK official`.

If the target product is a trustworthy long-term assistant memory, prioritize `PMS-Bench`.

If the target product is K3M as a perceptual memory backend, the recommended default is:

- use `AAAK official` as the retrieval baseline
- use `PMS-Bench` as the primary alignment benchmark

## Current Bottom Line
`AAAK official` answers:

`Can the system retrieve efficiently from realistic conversational haystacks?`

`PMS-Bench` answers:

`Can the system preserve the user's actual remembered structure under compression?`

K3M should be positioned as a system that ultimately aims to do both, but whose deepest philosophical alignment is with the second question.
