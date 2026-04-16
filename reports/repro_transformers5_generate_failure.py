#!/usr/bin/env python3
"""Minimal black-box reproducer for the transformers 5.x generate failure."""

from __future__ import annotations

import importlib.util
from importlib.machinery import SourceFileLoader

import torch
import transformers


TARGET_PATH = "/home/ubuntu/.cursor/projects/workspace/uploads/AgentMemorySystem.md"


def load_target_module(path: str):
    loader = SourceFileLoader("agent_memory_system_repro", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"transformers={transformers.__version__}")

    module = load_target_module(TARGET_PATH)
    torch.manual_seed(42)
    model = module.MemLLM(module.Cfg())
    model.load("gpt2")
    print(model.generate("Hello", mt=15, greedy=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
