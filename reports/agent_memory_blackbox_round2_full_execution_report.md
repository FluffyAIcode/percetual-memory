# AgentMemorySystem 第二轮 full 黑盒测试执行报告

## 1. 执行说明

第二轮 full 结果由以下两部分聚合而成：

- 已执行的 representative 场景 4 个
- 本轮补跑的 full-only 场景 5 个

这样覆盖了第二轮矩阵中的全部 9 个场景。

## 2. 环境

- Python: 3.12.3
- Torch: 2.11.0+cu130
- Transformers: 4.57.6
- Model: gpt2

## 3. 汇总结果

- PASS: 7
- WARN: 2
- FAIL: 0
- 聚合总耗时: `851.07s`

## 4. 分场景结果

### R2-STRESS-01 repeated write/generate pressure

- 类型: stress
- 状态: PASS
- 耗时: `70.97s`
- 结论: Completed repeated write/generate pressure loop without crash.

```json
{
  "avg_gate": 0.564429,
  "rounds": 2,
  "sample_outputs": {
    "cooking": "The chef prepared culinary chef increased pastry chefs reduced mission of reduce pasta cook and musicals, the team.\n a",
    "finance": "The market outlook musical market the mission of increased a, and reduce in that is an-\n. to be all",
    "music": "The piano performance musical music the mission of increased, and reduce a- or in this reduced to be more that is",
    "space": "The space telescope planets around musicals mission- the and increased. reduce team of a, that's\n in this"
  },
  "total_generations": 8,
  "total_writes": 24
}
```

### R2-STRESS-02 repeated save/load pressure

- 类型: stress
- 状态: PASS
- 耗时: `194.67s`
- 结论: Repeated save/load cycles preserved externally valid generation behavior.

```json
{
  "avg_gate": 0.564429,
  "cycle_outputs": [
    {
      "cooking": "The chef prepared culinary chef increased pastry chefs reduced mission of reduce pasta cook and musicals, the team.\n a",
      "finance": "The market outlook musical market the mission of increased a, and reduce in that's reduced to be more\n- or",
      "music": "The piano performance musical music the mission of increased, and reduce a- or in this reduced to be more that is",
      "space": "The space telescope planets orbit around musical- the team, and mission of increased to reduce a. in that's\n"
    },
    {
      "cooking": "The chef prepared culinary chef increased pastry chefs reduced mission of reduce pasta cook and musicals, the team.\n a",
      "finance": "The market outlook musical market the mission of increased a, and reduce in that is an-\n. to be all",
      "music": "The piano performance musical music the mission of increased, and reduce a- or in this reduced to be more that is",
      "space": "The space telescope planets around musicals mission- the and increased. reduce team of a, that's\n in this"
    }
  ],
  "cycles": 2
}
```

### R2-LONG-01 long memory write grounding

- 类型: long-text
- 状态: PASS
- 耗时: `63.66s`
- 结论: Long memory write remained usable for downstream generation.

```json
{
  "gate": 0.293301,
  "keyword_hits": [
    "concert",
    "music",
    "musical",
    "piano",
    "practice",
    "theory"
  ],
  "long_text_chars": 4099,
  "output": "The piano performance piano Music Practice music practice musical theory concerting hard, night and hours of the morning hour\n a- in this evening The"
}
```

### R2-LONG-02 long prompt resilience

- 类型: long-text
- 状态: PASS
- 耗时: `68.72s`
- 结论: Long prompt generation completed without crashing.

```json
{
  "output_chars": 4186,
  "prompt_chars": 4139
}
```

### R2-CROSS-01 dual-domain contamination diagnostic

- 类型: cross-domain
- 状态: WARN
- 耗时: `65.59s`
- 结论: Dual-domain run showed contamination or weak own-domain separation.

```json
{
  "music_foreign_hits": [
    "mission",
    "planet"
  ],
  "music_output": "The piano performance musical music the mission of a, and planets-\n. in that is an all other's to",
  "music_own_hits": [
    "music",
    "musical"
  ],
  "space_foreign_hits": [
    "music",
    "musical",
    "theory"
  ],
  "space_output": "The space telescope planets beyond the musical theory of a- and mission\n, in that is an all other. to",
  "space_own_hits": [
    "mission",
    "planet"
  ]
}
```

### R2-CROSS-02 four-domain contamination matrix

- 类型: cross-domain
- 状态: WARN
- 耗时: `68.97s`
- 结论: Four-way contamination matrix detected cross-domain bleed or weak own-domain signal.

```json
{
  "matrix": {
    "cooking": {
      "cooking": [
        "chef",
        "pasta"
      ],
      "finance": [],
      "music": [
        "music",
        "musical"
      ],
      "space": [
        "mission"
      ]
    },
    "finance": {
      "cooking": [],
      "finance": [
        "market"
      ],
      "music": [
        "music",
        "musical"
      ],
      "space": [
        "mission"
      ]
    },
    "music": {
      "cooking": [],
      "finance": [],
      "music": [
        "music",
        "musical"
      ],
      "space": [
        "mission"
      ]
    },
    "space": {
      "cooking": [],
      "finance": [],
      "music": [
        "music",
        "musical"
      ],
      "space": [
        "mission",
        "orbit",
        "planet"
      ]
    }
  }
}
```

### R2-STABLE-01 fresh instance determinism

- 类型: stability
- 状态: PASS
- 耗时: `121.20s`
- 结论: Fresh seeded instances were exactly deterministic under greedy generation.

```json
{
  "output": "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has",
  "prompt": "The piano performance"
}
```

### R2-STABLE-02 same instance repeatability

- 类型: stability
- 状态: PASS
- 耗时: `65.30s`
- 结论: Repeated greedy calls on the same instance were identical.

```json
{
  "outputs": [
    "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has",
    "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has",
    "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has"
  ]
}
```

### R2-STABLE-03 save/load exactness

- 类型: stability
- 状态: PASS
- 耗时: `131.99s`
- 结论: Greedy output stayed exactly stable across save/load roundtrip.

```json
{
  "output": "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has"
}
```

## 5. 关键发现

- 第二轮 full 没有阻断性 FAIL。
- WARN 全部集中在跨域污染场景。
- 四域污染矩阵显示 music/space 仍可保留一定 own-domain 信号，但 finance/cooking 更容易被其它域压制。
