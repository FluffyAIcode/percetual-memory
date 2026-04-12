# perceptial-memory

Standalone perceptual memory module extracted for later `agentmate` integration.

## What is included

- `src/`
  - `constraint-edge fixed` benchmark runner
  - PMS-Bench evaluator, schema, and prediction adapter
  - supporting state-extraction and support-construction modules
- `data/`
  - current `PMS-Bench v5` curated benchmark set
  - comparison results for `AAAK official` and `constraint-edge fixed`
- `docs/`
  - formal benchmark positioning
  - PMS-Bench comparison report
  - perceptual memory benchmark specification
  - updated K3M arXiv draft source and PDF

## Positioning

This repository packages the part of the K3M work that is most directly about perceptual memory:

- structured constraint recall
- turning-point and causal recall
- support-first memory compression
- benchmark framing for `retrieval realism` vs `perceptual-memory realism`

## Current benchmark takeaway

- `AAAK official` is stronger on compression-oriented retrieval realism.
- `constraint-edge fixed` is stronger on `PMS-Bench`, where the task is to preserve user constraints, pivots, and structured memory state under compression.

## Main documents

- `docs/benchmark_positioning_k3m_pms_vs_aaak.md`
- `docs/pms_bench_compare_report_aaak_official_vs_constraint_edge.md`
- `docs/perceptual_memory_module_overview.md`
