# AgentMemorySystem 第三轮黑盒测试矩阵设计

## 1. 目标

第三轮黑盒测试在前两轮基础上继续扩展三类能力面：

1. 边界输入测试
2. 异常输入测试
3. 性能 / 时延基线

仍坚持黑盒边界：

- 不修改被测实现
- 不使用 mock
- 不读取内部 memory tree、缓存或私有状态
- 不调用源码内置 `test()` / `test_*()` 自测函数
- 不把断言写成固定完整文本匹配

## 2. 公开调用面

第三轮只使用以下公开接口：

- `MemLLM.load()`
- `MemLLM.write()`
- `MemLLM.generate()`
- `MemLLM.save_memory()`
- `MemLLM.load_memory()`

## 3. 状态语义

- `PASS`：满足场景验收规则
- `WARN`：场景完成但暴露风险、边界不稳或性能超出经验阈值
- `FAIL`：违反硬性要求或出现未接受的外部异常

## 4. 第三轮测试矩阵

| ID | 类型 | 场景 | 性质 | 核心刺激 | 主要观察指标 | 验收规则 |
|---|---|---|---|---|---|---|
| R3-BOUNDARY-01 | 边界输入 | empty prompt generate | diagnostic | `generate("")` | 是否报错、是否产生输出 | 不崩溃为 PASS；异常为 WARN |
| R3-BOUNDARY-02 | 边界输入 | punctuation-only prompt | diagnostic | 只含标点的 prompt | 前缀保持、输出扩展 | 完成生成为 PASS |
| R3-BOUNDARY-03 | 边界输入 | whitespace-heavy prompt | diagnostic | 空格/换行密集 prompt | 是否报错、输出长度 | 完成生成为 PASS |
| R3-BOUNDARY-04 | 边界输入 | minimal memory write | gating | 极短文本写入 | `write()` 是否返回有限 gate | 无异常且 gate 有限 |
| R3-EXC-01 | 异常输入 | non-string write input | gating | `write(None)` / `write(123)` | 是否抛出外部异常 | 必须抛出异常且进程不挂死 |
| R3-EXC-02 | 异常输入 | non-string generate input | gating | `generate(None)` / `generate(123)` | 是否抛出外部异常 | 必须抛出异常且进程不挂死 |
| R3-EXC-03 | 异常输入 | missing memory load path | gating | `load_memory("/tmp/not-found.pt")` | 是否抛出文件错误 | 必须抛出异常 |
| R3-EXC-04 | 异常输入 | invalid memory file load | gating | 对随机文本文件执行 `load_memory()` | 是否抛出解析异常 | 必须抛出异常 |
| R3-PERF-01 | 性能基线 | cold load baseline | diagnostic | 全新实例 `load("gpt2")` | 加载耗时 | 记录基线；超经验阈值记 WARN |
| R3-PERF-02 | 性能基线 | write latency baseline | diagnostic | 连续写入 3 条记忆 | 单次与平均耗时 | 记录基线；异常慢记 WARN |
| R3-PERF-03 | 性能基线 | generate latency baseline | diagnostic | 空记忆和有记忆两种生成 | 生成耗时 | 记录基线；异常慢记 WARN |
| R3-PERF-04 | 性能基线 | save/load roundtrip latency | diagnostic | 保存+读取记忆 | save/load 各自耗时 | 记录基线；异常慢记 WARN |

## 5. 设计意图

### 5.1 边界输入

边界输入不是为了证明模型“语义上合理”，而是为了观察：

- 接口是否在非常规输入下崩溃
- 是否破坏 prompt 前缀约定
- 是否出现空输出、死循环或明显外部异常

### 5.2 异常输入

第三轮把“错误输入是否被外部稳定处理”纳入黑盒验证范围。

这里不要求实现必须给出优雅错误文案，但要求：

- 异常是**可观察、可终止、可定位**的
- 不应导致进程卡死
- 不应悄悄吞错并给出伪正常结果

### 5.3 性能 / 时延基线

性能场景重点不是追求绝对快，而是建立一条黑盒基线，回答：

- 冷启动大约多慢
- 单次写入大约多慢
- 生成大约多慢
- save/load 回环大约多慢

这些数据可为后续版本做对比回归。

## 6. 可执行资产

- 第三轮 runner：
  `/workspace/blackbox_test_agent_memory_round3.py`

## 7. 推荐执行命令

代表集：

```bash
python3 /workspace/blackbox_test_agent_memory_round3.py --suite representative
```

全量集：

```bash
python3 /workspace/blackbox_test_agent_memory_round3.py --suite full
```

导出 JSON：

```bash
python3 /workspace/blackbox_test_agent_memory_round3.py \
  --suite full \
  --json-out /workspace/reports/agent_memory_blackbox_round3_results.json
```
