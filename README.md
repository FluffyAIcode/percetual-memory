# perceptial-memory

Standalone perceptual memory module extracted from the broader K3M work for later `agentmate` integration.

## Overview

This repository focuses on one specific question:

Can a compressed memory system preserve the user structures that matter in practice, not only retrieve vaguely related past sessions?

The current module centers on:

- structured constraint recall
- turning-point and causal recall
- support-first memory compression
- benchmark framing for `retrieval realism` vs `perceptual-memory realism`

## Repository Layout

- `src/`
  Core benchmark runners, evaluator, schema, and support-construction code.
- `data/`
  Current `PMS-Bench v5` benchmark set and comparison result artifacts.
- `docs/`
  Positioning notes, comparison reports, benchmark spec, and paper draft materials.

## Main Components

- `src/pms_constraint_edge_benchmark.py`
  Runs the `constraint-edge fixed` perceptual-memory path.
- `src/perceptual_memory_benchmark.py`
  Evaluates predictions under PMS-Bench.
- `src/perceptual_memory_benchmark_schema.json`
  Runtime schema for benchmark items, predictions, sources, and results.
- `src/pms_prediction_adapter.py`
  Converts source records into benchmark prediction documents.
- `src/pms_sample_dataset_builder.py`
  Builds curated PMS-Bench benchmark artifacts.

## Current Benchmark Takeaway

- `AAAK official` is stronger on compression-oriented retrieval realism.
- `constraint-edge fixed` is stronger on `PMS-Bench`, where the task is to preserve user constraints, pivots, and structured memory state under compression.

The most important documents for that comparison are:

- `docs/benchmark_positioning_k3m_pms_vs_aaak.md`
- `docs/pms_bench_compare_report_aaak_official_vs_constraint_edge.md`
- `docs/perceptual_memory_module_overview.md`

## Quickstart

See `docs/quickstart.md` for the minimal workflow to:

- inspect the benchmark dataset
- run `constraint-edge fixed`
- evaluate outputs with PMS-Bench
- interpret the resulting metrics

## Positioning

This repository is intended as a standalone perceptual memory package rather than a full general-purpose retrieval stack.

In short:

- `AAAK official` measures retrieval realism
- `PMS-Bench` measures perceptual-memory realism
- this module is primarily aligned with the second
