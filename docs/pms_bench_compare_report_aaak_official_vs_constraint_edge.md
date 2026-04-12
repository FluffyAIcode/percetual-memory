# PMS-Bench AAAK Official vs Constraint-Edge Report

## Setup
- Dataset: `PMS-Bench v5` curated seed set (`20` items).
- AAAK path: official MemPalace retrieval via `build_palace_and_retrieve_aaak()`.
- Constraint-edge path: `LLMConstraintEdgeAddressEngram` with fallback state extraction, query-guided constraint canonicalization, cue-aware reranking, and support-archive byte accounting.
- Prediction/evaluation artifacts:
  - `pms_aaak_official_v5_results.json`
  - `pms_constraint_edge_v5_fixed_results.json`

## Main Comparison

| Method | PMS score | Retrieval | Constraint fidelity | Perceptual recovery | Recall@1 | Recall@3 | Recall@5 | MRR | Avg raw bytes | Avg bytes used | Avg compression | Support ratio | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAAK official | 0.4078 | 0.5639 | 0.0000 | 0.2832 | 0.4500 | 0.6000 | 0.6500 | 0.5558 | 1209.0 | 81.2 | 14.8924x | 0.0000 | 2991.69 ms |
| constraint-edge fixed | 0.7159 | 0.8858 | 1.0000 | 0.3709 | 0.7500 | 0.9500 | 1.0000 | 0.8433 | 1458.8 | 366.1 | 3.9820x | 0.5872 | 3.20 ms |

## Constraint Recall Focus

| Method | Allowed F1 | Forbidden F1 | Required F1 | Actual-next accuracy | Status accuracy |
|---|---:|---:|---:|---:|---:|
| AAAK official | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| constraint-edge fixed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Reference Runs

| Method | PMS score | Retrieval | Constraint fidelity | Recall@1 | Recall@3 | MRR | Avg bytes used | Support ratio | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AAAK lite | 0.2294 | 0.0428 | 0.0000 | 0.0000 | 0.0500 | 0.0711 | 108.0 | 0.0000 | 0.47 ms |
| constraint-edge old | 0.5560 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1551.4 | 0.4255 | 2.42 ms |

## Interpretation
- `AAAK official` remains the strongest pure compression path in this comparison, but on `PMS-Bench` it does not recover structured constraint objects.
- `constraint-edge fixed` gives up some compression ratio relative to AAAK official, but wins on total `pms_score`, retrieval, constraint fidelity, and latency.
- The key change versus the old constraint-edge run is not better top-k retrieval alone; it is aligning retrieved session content with benchmark-native constraint labels and then measuring a real support archive instead of raw-text placeholder bytes.

## Caveats
- `PMS-Bench` emphasizes perceptual/structural recovery rather than only top-k session retrieval, so its ranking can differ from LongMemEval-style leaderboard ordering.
- `archive_support_ratio` is meaningful for constraint-edge because it has an explicit support representation; it is `0` for AAAK official here because the official retrieval path does not expose an analogous address-edge support archive.
- The AAAK official query path is much slower in this local setup because each PMS item calls the full official retrieval pipeline rather than a prebuilt offline cache.
