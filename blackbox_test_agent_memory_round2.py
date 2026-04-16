#!/usr/bin/env python3
"""Second-round black-box matrix runner for AgentMemorySystem.

This runner extends the first-round smoke tests with broader black-box
coverage:

- stress scenarios
- long-text scenarios
- cross-domain contamination diagnostics
- stability scenarios

The runner still treats the uploaded implementation as an opaque component and
only uses its public runtime behavior:

- MemLLM.load()
- MemLLM.write()
- MemLLM.generate()
- MemLLM.save_memory()
- MemLLM.load_memory()

Status semantics:
- PASS: scenario met its acceptance rule
- WARN: scenario finished, but a diagnostic risk was observed
- FAIL: scenario violated a gating requirement or raised an unexpected error
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib.machinery import SourceFileLoader
from typing import Callable

import torch
import transformers


TARGET_PATH = "/home/ubuntu/.cursor/projects/workspace/uploads/AgentMemorySystem.md"
MODEL_NAME = "gpt2"
DEFAULT_SEED = 42

PROMPTS = {
    "music": "The piano performance",
    "space": "The space telescope",
    "finance": "The market outlook",
    "cooking": "The chef prepared",
}

CORPORA = {
    "music": [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "She studied music theory and harmonic progression at the conservatory.",
        "The orchestra rehearsed the symphony before the evening concert.",
    ],
    "space": [
        "Astronauts trained for the Mars mission in simulated zero gravity.",
        "The telescope revealed distant galaxies beyond the Milky Way.",
        "Mission control tracked the spacecraft during orbital insertion.",
    ],
    "finance": [
        "Investors monitored inflation data before the central bank meeting.",
        "The portfolio manager reduced exposure to volatile growth stocks.",
        "Quarterly earnings guidance shifted sentiment across the market.",
    ],
    "cooking": [
        "The chef reduced the sauce slowly before plating the duck.",
        "Fresh basil and olive oil brightened the pasta at the finish.",
        "The pastry team tempered chocolate for the dessert service.",
    ],
}

KEYWORDS = {
    "music": {
        "music",
        "musical",
        "violin",
        "concert",
        "symphony",
        "guitar",
        "practice",
        "practicing",
        "piano",
        "theory",
    },
    "space": {
        "space",
        "telescope",
        "galax",
        "orbit",
        "orbital",
        "mars",
        "mission",
        "astronaut",
        "spacecraft",
        "planet",
    },
    "finance": {
        "market",
        "stocks",
        "portfolio",
        "inflation",
        "bank",
        "earnings",
        "investor",
        "trading",
        "equity",
        "sentiment",
    },
    "cooking": {
        "chef",
        "sauce",
        "pasta",
        "dessert",
        "olive",
        "basil",
        "plating",
        "chocolate",
        "kitchen",
        "roasted",
    },
}


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
    loader = SourceFileLoader("agent_memory_system_round2", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def build_model(module, seed: int = DEFAULT_SEED):
    torch.manual_seed(seed)
    cfg = module.Cfg()
    model = module.MemLLM(cfg)
    model.load(MODEL_NAME)
    return model


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def keyword_hits(text: str, keywords: set[str]) -> list[str]:
    lowered = text.lower()
    return sorted(keyword for keyword in keywords if keyword in lowered)


def continuation(prompt: str, output: str) -> str:
    ensure(output.startswith(prompt), f"Output does not preserve prompt prefix: {output!r}")
    return output[len(prompt) :]


def mean_gate(values: list[float]) -> float:
    ensure(values, "No gate values were collected")
    return float(sum(values) / len(values))


def stable_write(model, texts: list[str]) -> list[float]:
    gates: list[float] = []
    for text in texts:
        stored, gate_vals = model.write(text, training_mode=True)
        ensure(stored == 1, f"Expected one stored memory for training_mode=True, got {stored}")
        ensure(len(gate_vals) == 1, f"Expected one gate value, got {gate_vals}")
        ensure(math.isfinite(gate_vals[0]), f"Gate value is not finite: {gate_vals[0]}")
        gates.extend(gate_vals)
    return gates


def run_scenario(module, scenario: ScenarioDef) -> ScenarioResult:
    start = time.perf_counter()
    try:
        status, summary, metrics = scenario.runner(module)
    except AssertionError as exc:
        status = "FAIL"
        summary = str(exc)
        metrics = {}
    except Exception as exc:  # pragma: no cover - surfaced in terminal output
        status = "FAIL"
        summary = f"{type(exc).__name__}: {exc}"
        metrics = {}
    duration_s = time.perf_counter() - start
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


def scenario_stress_write_generate(module) -> tuple[str, str, dict]:
    model = build_model(module)
    gate_values: list[float] = []
    prompt_outputs: dict[str, str] = {}
    rounds = 2
    for _ in range(rounds):
        for domain in ("music", "space", "finance", "cooking"):
            gate_values.extend(stable_write(model, CORPORA[domain]))
        for domain, prompt in PROMPTS.items():
            output = model.generate(prompt, mt=20, greedy=True)
            ensure(isinstance(output, str), f"generate() did not return a string for {domain}")
            ensure(len(output) > len(prompt), f"generate() did not extend prompt for {domain}")
            continuation(prompt, output)
            prompt_outputs[domain] = output
    return (
        "PASS",
        "Completed repeated write/generate pressure loop without crash.",
        {
            "rounds": rounds,
            "total_writes": rounds * sum(len(v) for v in CORPORA.values()),
            "total_generations": rounds * len(PROMPTS),
            "avg_gate": round(mean_gate(gate_values), 6),
            "sample_outputs": prompt_outputs,
        },
    )


def scenario_stress_save_load_cycles(module) -> tuple[str, str, dict]:
    model = build_model(module)
    gate_values: list[float] = []
    for domain in ("music", "space", "finance", "cooking"):
        gate_values.extend(stable_write(model, CORPORA[domain]))

    outputs_by_cycle: list[dict[str, str]] = []
    cycles = 2
    fd, memory_path = tempfile.mkstemp(prefix="agent-memory-round2-", suffix=".pt")
    os.close(fd)
    current = model
    try:
        for _ in range(cycles):
            current.save_memory(memory_path)
            ensure(os.path.getsize(memory_path) > 0, "save_memory() produced an empty file")
            reloaded = build_model(module)
            reloaded.load_memory(memory_path)
            cycle_outputs = {}
            for domain, prompt in PROMPTS.items():
                output = reloaded.generate(prompt, mt=20, greedy=True)
                continuation(prompt, output)
                cycle_outputs[domain] = output
            outputs_by_cycle.append(cycle_outputs)
            current = reloaded
    finally:
        if os.path.exists(memory_path):
            os.remove(memory_path)

    return (
        "PASS",
        "Repeated save/load cycles preserved externally valid generation behavior.",
        {
            "cycles": cycles,
            "avg_gate": round(mean_gate(gate_values), 6),
            "cycle_outputs": outputs_by_cycle,
        },
    )


def scenario_long_memory_write(module) -> tuple[str, str, dict]:
    model = build_model(module)
    long_text = " ".join(CORPORA["music"] * 20)
    stored, gates = model.write(long_text, training_mode=True)
    ensure(stored == 1, f"Expected one stored memory for long text, got {stored}")
    ensure(len(gates) == 1 and math.isfinite(gates[0]), f"Unexpected gate values: {gates}")

    prompt = PROMPTS["music"]
    output = model.generate(prompt, mt=25, greedy=True)
    cont = continuation(prompt, output)
    hits = keyword_hits(cont, KEYWORDS["music"])
    ensure(hits, f"Long-memory write did not yield music-domain hits: {cont!r}")
    return (
        "PASS",
        "Long memory write remained usable for downstream generation.",
        {
            "long_text_chars": len(long_text),
            "gate": round(gates[0], 6),
            "output": output,
            "keyword_hits": hits,
        },
    )


def scenario_long_prompt_resilience(module) -> tuple[str, str, dict]:
    model = build_model(module)
    prompt = ("The piano performance review discussed harmony and rhythm in detail. " * 60).strip()
    try:
        output = model.generate(prompt, mt=10, greedy=True)
        continuation(prompt, output)
        return (
            "PASS",
            "Long prompt generation completed without crashing.",
            {
                "prompt_chars": len(prompt),
                "output_chars": len(output),
            },
        )
    except Exception as exc:
        return (
            "WARN",
            "Long prompt generation raised an externally visible error.",
            {
                "prompt_chars": len(prompt),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )


def scenario_cross_domain_dual(module) -> tuple[str, str, dict]:
    model = build_model(module)
    stable_write(model, CORPORA["music"] + CORPORA["space"])

    music_output = model.generate(PROMPTS["music"], mt=20, greedy=True)
    space_output = model.generate(PROMPTS["space"], mt=20, greedy=True)
    music_cont = continuation(PROMPTS["music"], music_output)
    space_cont = continuation(PROMPTS["space"], space_output)

    music_own = keyword_hits(music_cont, KEYWORDS["music"])
    music_foreign = keyword_hits(music_cont, KEYWORDS["space"])
    space_own = keyword_hits(space_cont, KEYWORDS["space"])
    space_foreign = keyword_hits(space_cont, KEYWORDS["music"])

    contamination_detected = bool(music_foreign or space_foreign)
    missing_own_signal = not music_own or not space_own
    status = "PASS"
    summary = "Dual-domain prompts preserved own-domain signal without obvious contamination."
    if contamination_detected or missing_own_signal:
        status = "WARN"
        summary = "Dual-domain run showed contamination or weak own-domain separation."

    return (
        status,
        summary,
        {
            "music_output": music_output,
            "space_output": space_output,
            "music_own_hits": music_own,
            "music_foreign_hits": music_foreign,
            "space_own_hits": space_own,
            "space_foreign_hits": space_foreign,
        },
    )


def scenario_cross_domain_fourway(module) -> tuple[str, str, dict]:
    model = build_model(module)
    for domain in ("music", "space", "finance", "cooking"):
        stable_write(model, CORPORA[domain])

    matrix: dict[str, dict[str, list[str]]] = {}
    warning = False
    for domain, prompt in PROMPTS.items():
        output = model.generate(prompt, mt=20, greedy=True)
        cont = continuation(prompt, output)
        row = {}
        for keyword_domain, keywords in KEYWORDS.items():
            row[keyword_domain] = keyword_hits(cont, keywords)
        matrix[domain] = row
        own = row[domain]
        foreign = {
            k: v for k, v in row.items() if k != domain and v
        }
        if not own or foreign:
            warning = True

    status = "WARN" if warning else "PASS"
    summary = (
        "Four-way contamination matrix detected cross-domain bleed or weak own-domain signal."
        if warning
        else "Four-way contamination matrix looked clean."
    )
    return status, summary, {"matrix": matrix}


def scenario_stability_fresh_instance(module) -> tuple[str, str, dict]:
    model_a = build_model(module, seed=DEFAULT_SEED)
    stable_write(model_a, CORPORA["music"])
    output_a = model_a.generate(PROMPTS["music"], mt=20, greedy=True)

    model_b = build_model(module, seed=DEFAULT_SEED)
    stable_write(model_b, CORPORA["music"])
    output_b = model_b.generate(PROMPTS["music"], mt=20, greedy=True)

    ensure(output_a == output_b, "Fresh seeded instances produced different greedy outputs")
    return (
        "PASS",
        "Fresh seeded instances were exactly deterministic under greedy generation.",
        {
            "prompt": PROMPTS["music"],
            "output": output_a,
        },
    )


def scenario_stability_same_instance(module) -> tuple[str, str, dict]:
    model = build_model(module)
    stable_write(model, CORPORA["music"])
    outputs = [model.generate(PROMPTS["music"], mt=20, greedy=True) for _ in range(3)]
    identical = outputs[0] == outputs[1] == outputs[2]
    return (
        "PASS" if identical else "WARN",
        "Repeated greedy calls on the same instance were identical." if identical else "Repeated greedy calls drifted on the same instance.",
        {
            "outputs": outputs,
        },
    )


def scenario_stability_roundtrip_exactness(module) -> tuple[str, str, dict]:
    model = build_model(module)
    stable_write(model, CORPORA["music"])
    baseline = model.generate(PROMPTS["music"], mt=20, greedy=True)

    fd, memory_path = tempfile.mkstemp(prefix="agent-memory-round2-exact-", suffix=".pt")
    os.close(fd)
    try:
        model.save_memory(memory_path)
        reloaded = build_model(module)
        reloaded.load_memory(memory_path)
        after_reload = reloaded.generate(PROMPTS["music"], mt=20, greedy=True)
    finally:
        if os.path.exists(memory_path):
            os.remove(memory_path)

    ensure(baseline == after_reload, "Greedy output changed after save/load roundtrip")
    return (
        "PASS",
        "Greedy output stayed exactly stable across save/load roundtrip.",
        {
            "output": baseline,
        },
    )


SCENARIOS = [
    ScenarioDef("R2-STRESS-01", "stress", "repeated write/generate pressure", "representative", scenario_stress_write_generate),
    ScenarioDef("R2-STRESS-02", "stress", "repeated save/load pressure", "full", scenario_stress_save_load_cycles),
    ScenarioDef("R2-LONG-01", "long-text", "long memory write grounding", "representative", scenario_long_memory_write),
    ScenarioDef("R2-LONG-02", "long-text", "long prompt resilience", "full", scenario_long_prompt_resilience),
    ScenarioDef("R2-CROSS-01", "cross-domain", "dual-domain contamination diagnostic", "representative", scenario_cross_domain_dual),
    ScenarioDef("R2-CROSS-02", "cross-domain", "four-domain contamination matrix", "full", scenario_cross_domain_fourway),
    ScenarioDef("R2-STABLE-01", "stability", "fresh instance determinism", "representative", scenario_stability_fresh_instance),
    ScenarioDef("R2-STABLE-02", "stability", "same instance repeatability", "full", scenario_stability_same_instance),
    ScenarioDef("R2-STABLE-03", "stability", "save/load exactness", "full", scenario_stability_roundtrip_exactness),
]


def select_scenarios(suite: str, scenario_ids: list[str] | None) -> list[ScenarioDef]:
    selected = []
    allowed_suites = {"representative"} if suite == "representative" else {"representative", "full"}
    id_filter = set(scenario_ids or [])
    for scenario in SCENARIOS:
        if scenario.suite not in allowed_suites:
            continue
        if id_filter and scenario.scenario_id not in id_filter:
            continue
        selected.append(scenario)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-round AgentMemorySystem black-box test matrix runner")
    parser.add_argument("--suite", choices=("representative", "full"), default="representative")
    parser.add_argument("--scenario", action="append", help="Optional scenario ID filter; may be supplied multiple times")
    parser.add_argument("--json-out", help="Optional path to write JSON results")
    args = parser.parse_args()

    ensure(os.path.exists(TARGET_PATH), f"Target file does not exist: {TARGET_PATH}")
    module = load_target_module(TARGET_PATH)
    scenarios = select_scenarios(args.suite, args.scenario)
    ensure(scenarios, "No scenarios selected")

    print("Second-round black-box matrix runner: AgentMemorySystem")
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
