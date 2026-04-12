# Installation

## Minimal Python Dependencies

Install the minimal runtime dependencies with:

```bash
pip install -r requirements.txt
```

Current minimal list:

- `numpy`
- `jsonschema`

## Path Expectations

The current module was extracted from a larger research workspace. Some optional code paths still expect:

- a cleaned LongMemEval-style dataset path supplied at runtime
- access to the official MemPalace dialect when using the local `mempalace` wrapper

## Running the CLI

From the repository root:

```bash
python src/cli.py --help
```

Example commands:

```bash
python src/cli.py run-constraint-edge --help
python src/cli.py adapt-predictions --help
python src/cli.py evaluate --help
```

## Integration Notes

For `agentmate` integration, the main useful entry points are:

- `src/pms_constraint_edge_benchmark.py`
- `src/pms_prediction_adapter.py`
- `src/perceptual_memory_benchmark.py`
- `src/cli.py`
