from __future__ import annotations

import sys


COMMANDS = {
    "build-items": "pms_sample_dataset_builder",
    "run-constraint-edge": "pms_constraint_edge_benchmark",
    "adapt-predictions": "pms_prediction_adapter",
    "evaluate": "perceptual_memory_benchmark",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("perceptial-memory CLI")
        print("")
        print("Commands:")
        for command, module_name in COMMANDS.items():
            print(f"  {command:<20} -> {module_name}.py")
        return 0

    command = sys.argv[1]
    module_name = COMMANDS.get(command)
    if module_name is None:
        print(f"Unknown command: {command}", file=sys.stderr)
        return 1

    sys.argv = [f"{module_name}.py", *sys.argv[2:]]
    module = __import__(module_name)
    if not hasattr(module, "main"):
        print(f"Module {module_name} does not expose main()", file=sys.stderr)
        return 1
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
