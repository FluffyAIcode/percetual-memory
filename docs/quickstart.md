# Quickstart

## Purpose

This quickstart covers the smallest useful workflow for the perceptial-memory module:

1. inspect the benchmark data
2. run the `constraint-edge fixed` path
3. evaluate with PMS-Bench
4. read the output metrics

## Files

- Benchmark items:
  `data/pms_bench_curated_seed_20_v5.json`
- Main runner:
  `src/pms_constraint_edge_benchmark.py`
- Adapter:
  `src/pms_prediction_adapter.py`
- Evaluator:
  `src/perceptual_memory_benchmark.py`
- Schema:
  `src/perceptual_memory_benchmark_schema.json`

## Minimal Flow

### 1. Build source records

```bash
python src/pms_constraint_edge_benchmark.py \
  --items data/pms_bench_curated_seed_20_v5.json \
  --data /path/to/longmemeval_s_cleaned.json \
  --candidate-pool-size 5 \
  --llm-backend cursor-agent \
  --llm-timeout-seconds 8 \
  --llm-max-retries 0 \
  --force-fallback-extractor \
  --json-out pms_constraint_edge_source.json
```

### 2. Adapt source records into predictions

```bash
python src/pms_prediction_adapter.py \
  --items data/pms_bench_curated_seed_20_v5.json \
  --source pms_constraint_edge_source.json \
  --longmemeval-data /path/to/longmemeval_s_cleaned.json \
  --json-out pms_constraint_edge_predictions.json
```

### 3. Evaluate

```bash
python src/perceptual_memory_benchmark.py \
  --items data/pms_bench_curated_seed_20_v5.json \
  --predictions pms_constraint_edge_predictions.json \
  --json-out pms_constraint_edge_results.json
```

## What to Look At

Main summary fields:

- `pms_score`
- `retrieval`
- `constraint_fidelity`
- `perceptual_recovery`
- `avg_raw_bytes`
- `avg_bytes_used`
- `avg_compression_ratio`
- `archive_support_ratio`
- `avg_latency_ms`

## Interpretation

Use the metrics in two groups:

- Retrieval realism:
  `recall@k`, `mrr`, retrieval subscore
- Perceptual-memory realism:
  constraint fidelity, causal and turning metrics, PMS score

If the system retrieves the right session but fails constraint fidelity, it is acting more like a retrieval engine than a perceptual memory module.

## Related Docs

- `docs/benchmark_positioning_k3m_pms_vs_aaak.md`
- `docs/pms_bench_compare_report_aaak_official_vs_constraint_edge.md`
- `docs/perceptual_memory_module_overview.md`
