## Cursor Cloud specific instructions

### Project overview

Pure Python research codebase (`perceptial-memory`) for perceptual memory benchmarking and the K3M compression codec. No web services, databases, or Docker involved. All code lives in `src/`.

### Running code

All Python scripts must be run from `src/` (or with `src/` on `sys.path`) because modules import each other directly by filename (e.g. `from k3m_codec import K3MCodec`). Use `python3` — there is no `python` alias in the VM.

**CLI entrypoint:** `python3 src/cli.py --help`  
See `docs/installation.md` and `docs/quickstart.md` for full command reference.

### Key standalone modules (no external data needed)

| Module | How to run |
|---|---|
| `perceptual_memory_benchmark.py` | `cd src && python3 perceptual_memory_benchmark.py --items ../data/pms_bench_curated_seed_20_v5.json --predictions <predictions.json> --json-out results.json` |
| `k3m_codec.py` | Import and use `K3MCodec`, `make_synthetic_dataset`, `benchmark_codec` directly in Python |
| `constraint_edge_volume.py` | Import and use `ConstraintEdgeVolumeProjector` directly in Python |

### Modules requiring external data or services

- `pms_constraint_edge_benchmark.py`, `pms_prediction_adapter.py`: need the LongMemEval dataset (`longmemeval_s_cleaned.json` from the `v0.1` GitHub release).
- `benchmark_compare_k3m_aaak.py`: also needs LongMemEval dataset.
- `k3m_v2_balanced.py`, `engram_directional_nontriviality_test.py`: import `from mempalace.dialect import Dialect` which loads an external file via `MEMPALACE_OFFICIAL_DIALECT_PATH` env var. Will crash without it.
- LLM-backed extraction (`llm_state_extractor.py`): optionally needs `ANTHROPIC_API_KEY` or cursor-agent CLI. Use `--force-fallback-extractor` to bypass LLM calls.

### Gotchas

- There is no test suite or linter configured. Validation is done by running the benchmark modules.
- The `src/mempalace/dialect.py` wrapper unconditionally tries to load an external file at import time; any module that transitively imports it will fail without `MEMPALACE_OFFICIAL_DIALECT_PATH` set to a valid file.
