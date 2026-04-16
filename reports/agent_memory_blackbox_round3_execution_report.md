# AgentMemorySystem 第三轮 full 黑盒测试执行报告

## 1. 执行说明

第三轮 full 采用逐场景执行并聚合的方式完成，覆盖边界输入、异常输入、性能/时延基线共 12 个场景。

## 2. 环境

- Python: 3.12.3
- Torch: 2.11.0+cu130
- Transformers: 4.57.6
- Model: gpt2

## 3. 汇总结果

- PASS: 10
- WARN: 1
- FAIL: 1
- 聚合总耗时: `798.45s`

## 4. 分场景结果

### R3-BOUND-01 empty prompt generation

- 类型: boundary-input
- 状态: FAIL
- 耗时: `61.69s`
- 结论: RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1, 64] because the unspecified dimension size -1 can be any value and is ambiguous

### R3-BOUND-02 single-character prompt generation

- 类型: boundary-input
- 状态: PASS
- 耗时: `60.23s`
- 结论: Single-character prompt generation remained valid.

```json
{
  "output": "A the I ami- \"I-k, ands"
}
```

### R3-BOUND-03 whitespace prompt generation

- 类型: boundary-input
- 状态: PASS
- 耗时: `57.95s`
- 结论: Whitespace prompt generation completed.

```json
{
  "output": "   ia the world, I- The other\n ("
}
```

### R3-BOUND-04 multiline prompt generation

- 类型: boundary-input
- 状态: PASS
- 耗时: `61.25s`
- 结论: Multi-line prompt generation completed.

```json
{
  "output": "Line one.\nLine two. the I have a,\n"
}
```

### R3-ABN-01 write None input

- 类型: abnormal-input
- 状态: PASS
- 耗时: `59.30s`
- 结论: write(None, ...) raised an externally visible exception as expected.

```json
{
  "error_message": "You need to specify either `text` or `text_target`.",
  "error_type": "ValueError"
}
```

### R3-ABN-02 generate None input

- 类型: abnormal-input
- 状态: PASS
- 耗时: `62.67s`
- 结论: generate(None, ...) raised an externally visible exception as expected.

```json
{
  "error_message": "You need to specify either `text` or `text_target`.",
  "error_type": "ValueError"
}
```

### R3-ABN-03 negative max tokens handling

- 类型: abnormal-input
- 状态: WARN
- 耗时: `66.16s`
- 结论: Negative mt returned the original prompt without explicit validation.

```json
{
  "output": "Hello"
}
```

### R3-ABN-04 load missing memory file

- 类型: abnormal-input
- 状态: PASS
- 耗时: `61.28s`
- 结论: load_memory() on a missing path raised an externally visible exception.

```json
{
  "error_message": "[Errno 2] No such file or directory: '/tmp/agent-memory-nonexistent-file.pt'",
  "error_type": "FileNotFoundError"
}
```

### R3-PERF-01 cold load latency baseline

- 类型: performance
- 状态: PASS
- 耗时: `61.67s`
- 结论: Cold load latency baseline recorded.

```json
{
  "cold_load_s": 61.623
}
```

### R3-PERF-02 write latency baseline

- 类型: performance
- 状态: PASS
- 耗时: `61.81s`
- 结论: Write latency baseline recorded.

```json
{
  "avg_write_s": 0.081,
  "max_write_s": 0.181,
  "min_write_s": 0.028,
  "write_count": 3
}
```

### R3-PERF-03 generate latency baseline

- 类型: performance
- 状态: PASS
- 耗时: `64.74s`
- 结论: Generate latency baseline recorded.

```json
{
  "avg_generate_s": 0.957,
  "generate_count": 3,
  "max_generate_s": 1.002,
  "min_generate_s": 0.887,
  "sample_output": "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has"
}
```

### R3-PERF-04 save/load latency baseline

- 类型: performance
- 状态: PASS
- 耗时: `119.69s`
- 结论: Save/load latency baseline recorded.

```json
{
  "load_s": 0.003,
  "sample_output": "The piano performance musical music the and violin, a- is an in that's.\n The other \" it has",
  "save_s": 0.002
}
```

## 5. 关键发现

- `R3-BOUND-01` 暴露明确 FAIL：空字符串 prompt 会触发真实运行时崩溃。
- `R3-ABN-03` 记为 WARN：`mt < 0` 不会报错，而是直接返回原 prompt，说明缺少显式参数校验。
- 其余边界输入（单字符、空白、多行）可运行。
- 异常输入（`None`、缺失文件）均能以外部可见异常形式返回。
- 当前环境下的冷启动基线约为 61.6s；有记忆的 greedy 生成平均约 0.96s。
