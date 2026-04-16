#!/usr/bin/env python3
"""Third-round black-box runner for AgentMemorySystem.

Focus areas:
- boundary inputs
- abnormal/exception-facing inputs
- performance and latency baselines

The target implementation is still treated as an opaque component. The runner
only uses the public runtime behavior exposed by:

- MemLLM.load()
- MemLLM.write()
- MemLLM.generate()
- MemLLM.save_memory()
- MemLLM.load_memory()
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from importlib.machinery import SourceFileLoader
from typing import Callable

import torch
import transformers


TARGET_PATH = "/home/ubuntu/.cursor/projects/workspace/uploads/AgentMemorySystem.md"
MODEL_NAME = "gpt2"
DEFAULT_SEED = 42

MUSIC_PROMPT = "The piano performance"
MUSIC_TEXTS = [
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "She studied music theory and harmonic progression at the conservatory.",
    "The orchestra rehearsed the symphony before the evening concert.",
]


@dataclass
class ScenarioDef:
    scenario_id: str
    category: str
    title: str
    suite: str
    runner: Callable[[object], tuple[str, str, dict]]


@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    title: str
    suite: str
    status: str
    duration_s: float
    summary: str
    metrics: dict


def load_target_module(path: str):
    loader = SourceFileLoader("agent_memory_system_round3", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_model(module, seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    cfg = module.Cfg()
    model = module.MemLLM(cfg)
    model.load(MODEL_NAME)
    return model


def continuation(prompt: str, output: str) -> str:
    ensure(output.startswith(prompt), f"Output does not preserve prompt prefix: {output!r}")
    return output[len(prompt) :]


def stable_music_seed(model) -> list[float]:
    gates: list[float] = []
    for text in MUSIC_TEXTS:
        stored, gate_vals = model.write(text, training_mode=True)
        ensure(stored == 1, f"Expected one stored memory, got {stored}")
        ensure(len(gate_vals) == 1, f"Expected one gate value, got {gate_vals}")
        ensure(math.isfinite(gate_vals[0]), f"Gate is not finite: {gate_vals[0]}")
        gates.extend(gate_vals)
    return gates


def run_scenario(module, scenario: ScenarioDef) -> ScenarioResult:
    started = time.perf_counter()
    try:
        status, summary, metrics = scenario.runner(module)
    except AssertionError as exc:
        status = "FAIL"
        summary = str(exc)
        metrics = {}
    except Exception as exc:  # pragma: no cover
        status = "FAIL"
        summary = f"{type(exc).__name__}: {exc}"
        metrics = {}
    duration_s = time.perf_counter() - started
    return ScenarioResult(
        scenario_id=scenario.scenario_id,
        category=scenario.category,
        title=scenario.title,
        suite=scenario.suite,
        status=status,
        duration_s=duration_s,
        summary=summary,
        metrics=metrics,
    )


def scenario_boundary_empty_prompt(module) -> tuple[str, str, dict]:
    model = build_model(module)
    output = model.generate("", mt=10, greedy=True)
    ensure(isinstance(output, str), "generate() did not return a string")
    ensure(len(output) > 0, "Empty prompt generation returned an empty string")
    return "PASS", "Empty prompt generation returned a non-empty string.", {"output": output}


def scenario_boundary_single_char_prompt(module) -> tuple[str, str, dict]:
    model = build_model(module)
    prompt = "A"
    output = model.generate(prompt, mt=12, greedy=True)
    continuation(prompt, output)
    return "PASS", "Single-character prompt generation remained valid.", {"output": output}


def scenario_boundary_whitespace_prompt(module) -> tuple[str, str, dict]:
    model = build_model(module)
    prompt = "   "
    output = model.generate(prompt, mt=10, greedy=True)
    ensure(isinstance(output, str), "generate() did not return a string")
    ensure(len(output) >= len(prompt), "Whitespace prompt output was shorter than prompt")
    return "PASS", "Whitespace prompt generation completed.", {"output": output}


def scenario_boundary_newline_prompt(module) -> tuple[str, str, dict]:
    model = build_model(module)
    prompt = "Line one.\nLine two."
    output = model.generate(prompt, mt=12, greedy=True)
    continuation(prompt, output)
    return "PASS", "Multi-line prompt generation completed.", {"output": output}


def scenario_abnormal_none_write(module) -> tuple[str, str, dict]:
    model = build_model(module)
    try:
        model.write(None, training_mode=True)  # type: ignore[arg-type]
    except Exception as exc:
        return (
            "PASS",
            "write(None, ...) raised an externally visible exception as expected.",
            {"error_type": type(exc).__name__, "error_message": str(exc)},
        )
    return "WARN", "write(None, ...) unexpectedly succeeded.", {}


def scenario_abnormal_none_generate(module) -> tuple[str, str, dict]:
    model = build_model(module)
    try:
        model.generate(None, mt=10, greedy=True)  # type: ignore[arg-type]
    except Exception as exc:
        return (
            "PASS",
            "generate(None, ...) raised an externally visible exception as expected.",
            {"error_type": type(exc).__name__, "error_message": str(exc)},
        )
    return "WARN", "generate(None, ...) unexpectedly succeeded.", {}


def scenario_abnormal_negative_mt(module) -> tuple[str, str, dict]:
    model = build_model(module)
    prompt = "Hello"
    output = model.generate(prompt, mt=-5, greedy=True)
    if output == prompt:
        return (
            "WARN",
            "Negative mt returned the original prompt without explicit validation.",
            {"output": output},
        )
    return (
        "WARN",
        "Negative mt did not raise and produced a nonstandard output.",
        {"output": output},
    )


def scenario_abnormal_invalid_load_memory(module) -> tuple[str, str, dict]:
    model = build_model(module)
    missing_path = "/tmp/agent-memory-nonexistent-file.pt"
    try:
        model.load_memory(missing_path)
    except Exception as exc:
        return (
            "PASS",
            "load_memory() on a missing path raised an externally visible exception.",
            {"error_type": type(exc).__name__, "error_message": str(exc)},
        )
    return "WARN", "load_memory() on a missing path unexpectedly succeeded.", {}


def scenario_perf_cold_load_baseline(module) -> tuple[str, str, dict]:
    started = time.perf_counter()
    model = build_model(module)
    elapsed = time.perf_counter() - started
    ensure(model is not None, "Model failed to initialize")
    return (
        "PASS",
        "Cold load latency baseline recorded.",
        {"cold_load_s": round(elapsed, 3)},
    )


def scenario_perf_write_latency(module) -> tuple[str, str, dict]:
    model = build_model(module)
    timings = []
    for text in MUSIC_TEXTS:
        started = time.perf_counter()
        stored, gates = model.write(text, training_mode=True)
        elapsed = time.perf_counter() - started
        ensure(stored == 1, f"Expected one stored memory, got {stored}")
        ensure(len(gates) == 1 and math.isfinite(gates[0]), f"Unexpected gate values: {gates}")
        timings.append(elapsed)
    return (
        "PASS",
        "Write latency baseline recorded.",
        {
            "write_count": len(timings),
            "avg_write_s": round(statistics.mean(timings), 3),
            "max_write_s": round(max(timings), 3),
            "min_write_s": round(min(timings), 3),
        },
    )


def scenario_perf_generate_latency(module) -> tuple[str, str, dict]:
    model = build_model(module)
    stable_music_seed(model)
    timings = []
    outputs = []
    for _ in range(3):
        started = time.perf_counter()
        output = model.generate(MUSIC_PROMPT, mt=20, greedy=True)
        elapsed = time.perf_counter() - started
        continuation(MUSIC_PROMPT, output)
        timings.append(elapsed)
        outputs.append(output)
    return (
        "PASS",
        "Generate latency baseline recorded.",
        {
            "generate_count": len(timings),
            "avg_generate_s": round(statistics.mean(timings), 3),
            "max_generate_s": round(max(timings), 3),
            "min_generate_s": round(min(timings), 3),
            "sample_output": outputs[0],
        },
    )


def scenario_perf_save_load_latency(module) -> tuple[str, str, dict]:
    import tempfile

    model = build_model(module)
    stable_music_seed(model)
    fd, memory_path = tempfile.mkstemp(prefix="agent-memory-round3-", suffix=".pt")
    os.close(fd)
    try:
        save_started = time.perf_counter()
        model.save_memory(memory_path)
        save_elapsed = time.perf_counter() - save_started
        ensure(os.path.getsize(memory_path) > 0, "save_memory() produced an empty file")

        reload_model = build_model(module)
        load_started = time.perf_counter()
        reload_model.load_memory(memory_path)
        load_elapsed = time.perf_counter() - load_started

        output = reload_model.generate(MUSIC_PROMPT, mt=20, greedy=True)
        continuation(MUSIC_PROMPT, output)
    finally:
        if os.path.exists(memory_path):
            os.remove(memory_path)

    return (
        "PASS",
        "Save/load latency baseline recorded.",
        {
            "save_s": round(save_elapsed, 3),
            "load_s": round(load_elapsed, 3),
            "sample_output": output,
        },
    )


SCENARIOS = [
    ScenarioDef("R3-BOUND-01", "boundary-input", "empty prompt generation", "representative", scenario_boundary_empty_prompt),
    ScenarioDef("R3-BOUND-02", "boundary-input", "single-character prompt generation", "full", scenario_boundary_single_char_prompt),
    ScenarioDef("R3-BOUND-03", "boundary-input", "whitespace prompt generation", "full", scenario_boundary_whitespace_prompt),
    ScenarioDef("R3-BOUND-04", "boundary-input", "multiline prompt generation", "full", scenario_boundary_newline_prompt),
    ScenarioDef("R3-ABN-01", "abnormal-input", "write None input", "representative", scenario_abnormal_none_write),
    ScenarioDef("R3-ABN-02", "abnormal-input", "generate None input", "full", scenario_abnormal_none_generate),
    ScenarioDef("R3-ABN-03", "abnormal-input", "negative max tokens handling", "full", scenario_abnormal_negative_mt),
    ScenarioDef("R3-ABN-04", "abnormal-input", "load missing memory file", "full", scenario_abnormal_invalid_load_memory),
    ScenarioDef("R3-PERF-01", "performance", "cold load latency baseline", "representative", scenario_perf_cold_load_baseline),
    ScenarioDef("R3-PERF-02", "performance", "write latency baseline", "full", scenario_perf_write_latency),
    ScenarioDef("R3-PERF-03", "performance", "generate latency baseline", "representative", scenario_perf_generate_latency),
    ScenarioDef("R3-PERF-04", "performance", "save/load latency baseline", "full", scenario_perf_save_load_latency),
]


def select_scenarios(suite: str, scenario_ids: list[str] | None) -> list[ScenarioDef]:
    allowed_suites = {"representative"} if suite == "representative" else {"representative", "full"}
    id_filter = set(scenario_ids or [])
    selected = []
    for scenario in SCENARIOS:
        if scenario.suite not in allowed_suites:
            continue
        if id_filter and scenario.scenario_id not in id_filter:
            continue
        selected.append(scenario)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Third-round AgentMemorySystem black-box runner")
    parser.add_argument("--suite", choices=("representative", "full"), default="representative")
    parser.add_argument("--scenario", action="append", help="Optional scenario ID filter; may be supplied multiple times")
    parser.add_argument("--json-out", help="Optional path to write JSON results")
    args = parser.parse_args()

    ensure(os.path.exists(TARGET_PATH), f"Target file does not exist: {TARGET_PATH}")
    module = load_target_module(TARGET_PATH)
    scenarios = select_scenarios(args.suite, args.scenario)
    ensure(scenarios, "No scenarios selected")

    print("Third-round black-box runner: AgentMemorySystem")
    print(f"Target file: {TARGET_PATH}")
    print(f"Suite: {args.suite}")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print("")

    started = time.perf_counter()
    results = [run_scenario(module, scenario) for scenario in scenarios]
    total_duration = time.perf_counter() - started

    pass_count = sum(1 for result in results if result.status == "PASS")
    warn_count = sum(1 for result in results if result.status == "WARN")
    fail_count = sum(1 for result in results if result.status == "FAIL")

    print("=" * 80)
    for result in results:
        print(
            f"[{result.status}] {result.scenario_id} | {result.category} | "
            f"{result.title} ({result.duration_s:.2f}s)"
        )
        print(result.summary)
        if result.metrics:
            print(json.dumps(result.metrics, indent=2, ensure_ascii=False, sort_keys=True))
        print("-" * 80)
    print(f"Summary: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count}")
    print(f"Total duration: {total_duration:.2f}s")

    if args.json_out:
        payload = {
            "target_path": TARGET_PATH,
            "suite": args.suite,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "total_duration_s": total_duration,
            "results": [asdict(result) for result in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)

    return 1 if fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
