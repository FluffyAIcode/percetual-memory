#!/usr/bin/env python3
"""Black-box test runner for the uploaded AgentMemorySystem implementation.

This runner intentionally treats the uploaded code as an opaque component and
only interacts with its public runtime behavior:

- MemLLM.load()
- MemLLM.write()
- MemLLM.generate()
- MemLLM.save_memory()
- MemLLM.load_memory()

It does not call private helpers, does not inspect internal memory state, and
does not use mocks.
"""

from __future__ import annotations

import importlib.util
import math
import os
import platform
import tempfile
import time
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader

import torch
import transformers


TARGET_PATH = "/home/ubuntu/.cursor/projects/workspace/uploads/AgentMemorySystem.md"
MODEL_NAME = "gpt2"
MUSIC_PROMPT = "The piano performance"
MUSIC_MEMORIES = [
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "She studied music theory and harmonic progression at the conservatory.",
    "The orchestra rehearsed the symphony before the evening concert.",
]
MUSIC_KEYWORDS = {
    "music",
    "musical",
    "violin",
    "concert",
    "symphony",
    "guitar",
    "practice",
    "practicing",
}


@dataclass
class CaseResult:
    case_id: str
    title: str
    passed: bool
    duration_s: float
    details: str


def load_target_module(path: str):
    loader = SourceFileLoader("agent_memory_system_under_test", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def keyword_hits(text: str, keywords: set[str]) -> list[str]:
    lowered = text.lower()
    return sorted(keyword for keyword in keywords if keyword in lowered)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def build_model(module):
    cfg = module.Cfg()
    model = module.MemLLM(cfg)
    model.load(MODEL_NAME)
    return model


def run_case(case_id: str, title: str, fn) -> CaseResult:
    start = time.perf_counter()
    try:
        details = fn()
        passed = True
    except Exception as exc:  # pragma: no cover - failure path is test output
        details = f"{type(exc).__name__}: {exc}"
        passed = False
    duration_s = time.perf_counter() - start
    return CaseResult(case_id, title, passed, duration_s, details)


def main() -> int:
    overall_start = time.perf_counter()
    print("Black-box test runner: AgentMemorySystem")
    print(f"Target file: {TARGET_PATH}")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print("")

    check(os.path.exists(TARGET_PATH), f"Target file does not exist: {TARGET_PATH}")
    module = load_target_module(TARGET_PATH)
    torch.manual_seed(42)
    model = build_model(module)
    results: list[CaseResult] = []
    state: dict[str, object] = {}

    def tc01_load_public_api() -> str:
        check(model is not None, "MemLLM.load() did not produce a model instance")
        return f"Loaded public model API successfully with {MODEL_NAME}"

    def tc02_generate_without_memory() -> str:
        output = model.generate("Hello", mt=15, greedy=True)
        check(isinstance(output, str), "generate() did not return a string")
        check(output.startswith("Hello"), "Generated text does not preserve the prompt prefix")
        check(len(output) > len("Hello"), "Generated text did not extend the prompt")
        return f"Output: {output!r}"

    def tc03_baseline_music_prompt_before_memory() -> str:
        output = model.generate(MUSIC_PROMPT, mt=20, greedy=True)
        continuation = output[len(MUSIC_PROMPT) :]
        hits = keyword_hits(continuation, MUSIC_KEYWORDS)
        state["baseline_music_output"] = output
        state["baseline_music_hits"] = hits
        check(output.startswith(MUSIC_PROMPT), "Prompt prefix was not preserved in the baseline run")
        return f"Baseline output: {output!r}\nBaseline keyword hits: {hits}"

    def tc04_write_and_ground_music_domain() -> str:
        write_lines = []
        for text in MUSIC_MEMORIES:
            stored, gates = model.write(text, training_mode=True)
            check(stored == 1, f"training_mode=True should store the input, got stored={stored}")
            check(len(gates) == 1, f"Expected exactly one gate value, got {gates}")
            check(math.isfinite(gates[0]), f"Gate value is not finite: {gates[0]}")
            write_lines.append(f"stored={stored}, gate={gates[0]:.6f}, text={text!r}")

        output = model.generate(MUSIC_PROMPT, mt=20, greedy=True)
        continuation = output[len(MUSIC_PROMPT) :]
        hits = keyword_hits(continuation, MUSIC_KEYWORDS)
        state["post_memory_music_output"] = output
        state["post_memory_music_hits"] = hits
        check(output.startswith(MUSIC_PROMPT), "Prompt prefix was not preserved after writing memory")
        check(hits, f"No music-domain grounding detected in continuation: {continuation!r}")
        return "\n".join(write_lines + [f"Output: {output!r}", f"Keyword hits: {hits}"])

    def tc05_memory_improves_domain_signal() -> str:
        baseline_hits = state.get("baseline_music_hits")
        post_hits = state.get("post_memory_music_hits")
        check(isinstance(baseline_hits, list), "Baseline music output was not recorded")
        check(isinstance(post_hits, list), "Post-memory music output was not recorded")
        check(
            len(post_hits) > len(baseline_hits),
            f"Music-domain signal did not improve: baseline={baseline_hits}, post={post_hits}",
        )
        return (
            f"Baseline hits: {baseline_hits}\n"
            f"Post-memory hits: {post_hits}\n"
            f"Baseline output: {state['baseline_music_output']!r}\n"
            f"Post-memory output: {state['post_memory_music_output']!r}"
        )

    def tc06_save_load_roundtrip() -> str:
        fd, memory_path = tempfile.mkstemp(prefix="agent-memory-", suffix=".pt")
        os.close(fd)
        try:
            model.save_memory(memory_path)
            check(os.path.exists(memory_path), "save_memory() did not create a file")
            file_size = os.path.getsize(memory_path)
            check(file_size > 0, "save_memory() created an empty file")

            torch.manual_seed(42)
            reloaded = build_model(module)
            reloaded.load_memory(memory_path)
            output = reloaded.generate(MUSIC_PROMPT, mt=20, greedy=True)
            continuation = output[len(MUSIC_PROMPT) :]
            hits = keyword_hits(continuation, MUSIC_KEYWORDS)
            check(output.startswith(MUSIC_PROMPT), "Reloaded model did not preserve the prompt prefix")
            check(hits, f"No music-domain grounding after reload: {continuation!r}")
            return (
                f"Saved file: {memory_path} ({file_size} bytes)\n"
                f"Output after reload: {output!r}\n"
                f"Keyword hits after reload: {hits}"
            )
        finally:
            if os.path.exists(memory_path):
                os.remove(memory_path)

    results.append(run_case("TC-01", "load public API", tc01_load_public_api))
    results.append(run_case("TC-02", "generate without memory", tc02_generate_without_memory))
    results.append(run_case("TC-03", "baseline music prompt before memory", tc03_baseline_music_prompt_before_memory))
    results.append(run_case("TC-04", "write memory and observe domain grounding", tc04_write_and_ground_music_domain))
    results.append(run_case("TC-05", "memory improves domain signal", tc05_memory_improves_domain_signal))
    results.append(run_case("TC-06", "save/load memory roundtrip", tc06_save_load_roundtrip))

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    total_duration = time.perf_counter() - overall_start

    print("=" * 72)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.case_id} - {result.title} ({result.duration_s:.2f}s)")
        print(result.details)
        print("-" * 72)
    print(f"Summary: {passed}/{len(results)} passed, {failed} failed")
    print(f"Total duration: {total_duration:.2f}s")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
