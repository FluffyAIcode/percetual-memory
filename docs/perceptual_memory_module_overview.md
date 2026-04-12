# Perceptual Memory Module Overview

## Goal

This module isolates the `perceptual memory` part of the broader K3M work so it can later be integrated into `agentmate` as a standalone memory subsystem.

The central question is not only:

- can a system retrieve a relevant past session?

It is also:

- can a compressed memory preserve the user's actual remembered structure?

That structure includes:

- preferences
- allowed and forbidden continuations
- required follow-up behavior
- causal explanations
- turning points and state shifts

## Core idea

The module follows a `support-first` philosophy.

Instead of treating memory as a dense transcript that must be searched everywhere, it tries to preserve a thinner structural support that carries the directions most useful for future recall. In the current textual implementation, those directions are represented as state transitions, constraint edges, and continuation structure rather than literal geometric lines.

## Main runtime path

The current strongest benchmarked path is `constraint-edge fixed`.

Its practical steps are:

1. Build compact state representations for candidate sessions.
2. Encode those states into address and edge supports.
3. Rank sessions using lexical evidence plus support overlap.
4. Canonicalize retrieved content into benchmark-native constraint objects.
5. Evaluate against `PMS-Bench`, which measures perceptual-memory fidelity rather than only generic retrieval.

## Benchmark interpretation

This module uses two complementary benchmark lenses:

- `AAAK official`
  - measures retrieval realism
  - asks whether the system can recover the right session efficiently from a realistic haystack
- `PMS-Bench`
  - measures perceptual-memory realism
  - asks whether the system preserves user-specific structure under compression

These benchmarks should not be read as contradictory. They reflect different product requirements.

## Current practical conclusion

- If the target product is a retrieval-heavy memory engine, `AAAK official` is a stronger realism target.
- If the target product is a trustworthy long-term assistant memory, `PMS-Bench` is the more aligned target.
- For K3M-inspired perceptual memory, `PMS-Bench` is the primary alignment benchmark.

## Included artifacts

- runnable source files for `constraint-edge` and PMS-Bench
- benchmark schema and evaluator
- current benchmark dataset and comparison outputs
- formal benchmark positioning notes
- updated K3M paper draft and PDF
