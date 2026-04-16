#!/usr/bin/env python3
"""Generate a black-box cross-domain contamination heatmap report.

The script uses only the public runtime behavior of the uploaded
AgentMemorySystem implementation:

- MemLLM.load()
- MemLLM.write()
- MemLLM.generate()

It writes all domain corpora into a single model instance, probes each domain
prompt, counts keyword hits by domain, and emits both JSON and Markdown
artifacts for heatmap-style inspection.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import time
from importlib.machinery import SourceFileLoader

import torch
import transformers


TARGET_PATH = "/home/ubuntu/.cursor/projects/workspace/uploads/AgentMemorySystem.md"
MODEL_NAME = "gpt2"
DEFAULT_SEED = 42
JSON_OUT = "/workspace/reports/agent_memory_cross_domain_heatmap.json"
MD_OUT = "/workspace/reports/agent_memory_cross_domain_heatmap.md"

PROMPTS = {
    "music": [
        "The piano performance",
        "The violin recital",
        "The music theory lesson",
    ],
    "space": [
        "The space telescope",
        "The Mars mission",
        "The orbital research team",
    ],
    "finance": [
        "The market outlook",
        "The portfolio manager",
        "The quarterly earnings call",
    ],
    "cooking": [
        "The chef prepared",
        "The pasta course",
        "The dessert service",
    ],
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

HEATMAP_SCALE = [
    (0, "0", "none"),
    (1, "1", "low"),
    (2, "2", "moderate"),
    (3, "3", "moderate"),
    (4, "4", "high"),
]


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_target_module(path: str):
    loader = SourceFileLoader("agent_memory_heatmap", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError(f"Unable to create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def build_model(module):
    torch.manual_seed(DEFAULT_SEED)
    model = module.MemLLM(module.Cfg())
    model.load(MODEL_NAME)
    return model


def keyword_hits(text: str, keywords: set[str]) -> list[str]:
    lowered = text.lower()
    return sorted(keyword for keyword in keywords if keyword in lowered)


def continuation(prompt: str, output: str) -> str:
    ensure(output.startswith(prompt), f"Output does not preserve prompt prefix: {output!r}")
    return output[len(prompt) :]


def stable_write(model, texts: list[str]) -> list[float]:
    gates: list[float] = []
    for text in texts:
        stored, gate_vals = model.write(text, training_mode=True)
        ensure(stored == 1, f"Expected one stored memory for training_mode=True, got {stored}")
        ensure(len(gate_vals) == 1, f"Expected one gate value, got {gate_vals}")
        gates.extend(gate_vals)
    return gates


def heat_symbol(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2"
    return "4"


def build_markdown(payload: dict) -> str:
    domains = list(PROMPTS)
    matrix = payload["matrix"]
    outputs = payload["outputs"]
    own_foreign = payload["own_vs_foreign"]
    prompt_rows = payload["prompt_rows"]

    lines = [
        "# AgentMemorySystem 跨域污染热图报告",
        "",
        "## 1. 执行环境",
        "",
        f"- Python: {payload['python']}",
        f"- Torch: {payload['torch']}",
        f"- Transformers: {payload['transformers']}",
        f"- Model: {payload['model_name']}",
        f"- 总耗时: `{payload['duration_s']:.2f}s`",
        "",
        "## 2. 说明",
        "",
        "该报告通过黑盒方式同时写入多个领域语料，再对各领域多个 prompt 变体进行生成，",
        "统计 continuation 中命中的各领域关键词数量。",
        "",
        "热图符号说明：",
        "",
        "- `0`: 无命中",
        "- `1`: 低污染/低命中",
        "- `2`: 中等",
        "- `4`: 高",
        "",
        "## 3. 放大版关键词命中计数矩阵",
        "",
    ]

    header = "| prompt variant\\\\keyword | " + " | ".join(domains) + " |"
    sep = "|" + "---|" * (len(domains) + 1)
    lines.extend([header, sep])
    for row_id in prompt_rows:
        row_cells = [row_id]
        for keyword_domain in domains:
            count = matrix[row_id][keyword_domain]["count"]
            symbol = heat_symbol(count)
            row_cells.append(f"{count} ({symbol})")
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.extend(
        [
            "",
            "## 4. 按领域汇总的 own-domain vs foreign-domain",
            "",
            "| prompt domain | own hits | foreign hits | foreign/own ratio | verdict |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for domain in domains:
        summary = own_foreign[domain]
        ratio = summary["foreign_to_own_ratio"]
        ratio_text = "inf" if ratio is None else f"{ratio:.2f}"
        lines.append(
            f"| {domain} | {summary['own_hits_count']} | {summary['foreign_hits_count']} | "
            f"{ratio_text} | {summary['verdict']} |"
        )

    lines.extend(["", "## 5. 各 prompt 变体生成样例", ""])
    for row_id in prompt_rows:
        domain = row_id.split("::", 1)[0]
        lines.extend(
            [
                f"### {row_id}",
                "",
                f"**Prompt**: `{outputs[row_id]['prompt']}`",
                "",
                f"**Output**:",
                "",
                "```text",
                outputs[row_id]["output"],
                "```",
                "",
                "**命中详情**:",
                "",
            ]
        )
        for keyword_domain in domains:
            hits = matrix[row_id][keyword_domain]["hits"]
            lines.append(f"- {keyword_domain}: {hits}")
        lines.append("")

    lines.extend(
        [
            "## 6. 结论",
            "",
            "如果 foreign hits 在多个 prompt 上持续显著非零，则说明系统存在跨域污染。",
            "如果 own hits 明显高于 foreign hits，则说明仍保留一定的领域接地能力。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ensure(os.path.exists(TARGET_PATH), f"Target file does not exist: {TARGET_PATH}")
    started = time.perf_counter()
    module = load_target_module(TARGET_PATH)
    model = build_model(module)
    gates: list[float] = []
    for domain in PROMPTS:
        gates.extend(stable_write(model, CORPORA[domain]))

    matrix: dict[str, dict[str, dict[str, object]]] = {}
    outputs: dict[str, dict[str, str]] = {}
    own_vs_foreign: dict[str, dict[str, object]] = {}
    prompt_rows: list[str] = []
    domains = list(PROMPTS)
    aggregate: dict[str, dict[str, int | float | None]] = {
        domain: {"own": 0, "foreign": 0} for domain in domains
    }
    for prompt_domain, prompts in PROMPTS.items():
        for idx, prompt in enumerate(prompts, start=1):
            row_id = f"{prompt_domain}::p{idx}"
            prompt_rows.append(row_id)
            output = model.generate(prompt, mt=20, greedy=True)
            cont = continuation(prompt, output)
            outputs[row_id] = {"prompt": prompt, "output": output}
            row = {}
            own_count = 0
            foreign_count = 0
            for keyword_domain in domains:
                hits = keyword_hits(cont, KEYWORDS[keyword_domain])
                count = len(hits)
                row[keyword_domain] = {
                    "hits": hits,
                    "count": count,
                }
                if keyword_domain == prompt_domain:
                    own_count += count
                else:
                    foreign_count += count
            matrix[row_id] = row
            aggregate[prompt_domain]["own"] += own_count
            aggregate[prompt_domain]["foreign"] += foreign_count

    for domain in domains:
        own_count = int(aggregate[domain]["own"])
        foreign_count = int(aggregate[domain]["foreign"])
        ratio = None if own_count == 0 else foreign_count / own_count
        verdict = "clean"
        if own_count == 0 or foreign_count >= own_count:
            verdict = "high-contamination"
        elif foreign_count > 0:
            verdict = "mixed"
        own_vs_foreign[domain] = {
            "own_hits_count": own_count,
            "foreign_hits_count": foreign_count,
            "foreign_to_own_ratio": ratio,
            "verdict": verdict,
        }

    duration = time.perf_counter() - started
    payload = {
        "target_path": TARGET_PATH,
        "model_name": MODEL_NAME,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "duration_s": duration,
        "avg_gate": sum(gates) / len(gates) if gates else None,
        "prompt_rows": prompt_rows,
        "matrix": matrix,
        "outputs": outputs,
        "own_vs_foreign": own_vs_foreign,
    }

    with open(JSON_OUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
    with open(MD_OUT, "w", encoding="utf-8") as handle:
        handle.write(build_markdown(payload))

    print("Cross-domain contamination heatmap generated.")
    print(f"JSON: {JSON_OUT}")
    print(f"Markdown: {MD_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
