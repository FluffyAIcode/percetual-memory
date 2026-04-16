# AgentMemorySystem 第二轮黑盒测试矩阵设计

## 1. 目标

在第一轮黑盒测试已经覆盖“加载、基础生成、记忆写入、持久化回环”的基础上，第二轮继续扩展以下四类风险面：

1. 压力测试
2. 长文本测试
3. 跨域污染测试
4. 稳定性测试

本轮仍坚持以下边界：

- 不修改被测实现
- 不使用 mock
- 不调用内部 `test()` / `test_*()` 自测函数
- 不读取内部 memory tree、缓存、私有状态
- 不把测试绑定到某个固定完整输出句子

## 2. 黑盒准则

### 2.1 允许观察的对象

只观察公开调用及其外部结果：

- `MemLLM.load()`
- `MemLLM.write()`
- `MemLLM.generate()`
- `MemLLM.save_memory()`
- `MemLLM.load_memory()`

### 2.2 判定方式

第二轮测试矩阵使用三档状态：

- `PASS`：满足场景既定验收规则
- `WARN`：场景可执行，但暴露出行为风险或质量问题
- `FAIL`：场景违反硬性要求，或发生未接受的外部异常

其中：

- **gating 场景**：以 `FAIL` 作为阻断结论
- **diagnostic 场景**：允许 `WARN`，用于暴露质量风险而非直接阻断

## 3. 领域测试数据

为了做跨域和污染观察，第二轮矩阵使用四组真实文本域：

- `music`
- `space`
- `finance`
- `cooking`

每个域使用 3 条真实自然语言样本文本，以及 1 条对应 prompt。

## 4. 测试矩阵

| ID | 类型 | 场景 | 性质 | 核心刺激 | 主要观察指标 | 验收规则 |
|---|---|---|---|---|---|---|
| R2-STRESS-01 | 压力 | repeated write/generate pressure | gating | 多轮写入 + 多 prompt 连续生成 | 是否崩溃、是否保留 prompt 前缀、输出是否扩展、gate 是否有限 | 全流程无崩溃且输出有效 |
| R2-STRESS-02 | 压力 | repeated save/load pressure | gating | 混合域写入后反复 save/load | 每轮回载后生成是否仍有效 | 每轮均可生成有效输出 |
| R2-LONG-01 | 长文本 | long memory write grounding | gating | 写入超长单条记忆文本 | 长文本写入后是否仍能对目标 prompt 产生领域词 | 输出有效且命中目标域关键词 |
| R2-LONG-02 | 长文本 | long prompt resilience | diagnostic | 超长 prompt 直接生成 | 是否能完成生成；若失败，失败类型是什么 | 不崩溃为 PASS；崩溃记 WARN |
| R2-CROSS-01 | 跨域污染 | dual-domain contamination diagnostic | diagnostic | 同时写入 music + space | own-domain hits、foreign hits | 无污染为 PASS；有污染或 own signal 弱则 WARN |
| R2-CROSS-02 | 跨域污染 | four-domain contamination matrix | diagnostic | 同时写入四域并逐 prompt 生成 | 四域命中矩阵、foreign hit 分布 | own hits 足且 foreign 低为 PASS，否则 WARN |
| R2-STABLE-01 | 稳定性 | fresh instance determinism | gating | 相同 seed、相同输入、不同新实例 | greedy 输出是否完全一致 | 必须完全一致 |
| R2-STABLE-02 | 稳定性 | same instance repeatability | diagnostic | 同一实例重复 greedy 生成 | 重复调用是否漂移 | 完全一致为 PASS；漂移则 WARN |
| R2-STABLE-03 | 稳定性 | save/load exactness | gating | 生成前后做 save/load 回环 | roundtrip 前后 greedy 输出是否完全一致 | 必须完全一致 |

## 5. 每类测试的设计意图

### 5.1 压力测试

第一轮只验证了小规模调用链路。第二轮压力测试关注：

- 连续写入后是否出现崩溃
- 连续生成后是否出现无输出或 prompt 破坏
- 多次 save/load 后是否出现状态损坏

这类测试更接近真实系统运行中的“记忆不断被写入和读取”的场景。

### 5.2 长文本测试

第一轮没有覆盖长样本。第二轮要验证：

- 单条超长 memory text 写入时是否还能正常使用
- 超长 prompt 输入是否会触发位置编码、上下文窗口或形状错误

其中长 prompt 场景被定义为 diagnostic，因为模型上下文本来可能存在边界限制；这类用例的重点是暴露边界行为，而不是强行把所有边界都定义为功能缺陷。

### 5.3 跨域污染测试

第一轮已经观察到混合域输入存在串扰风险。第二轮将其系统化，形成矩阵：

- 双域污染：快速定位最明显的串扰
- 四域污染：观察污染是否随域数量上升而恶化

注意：这类用例不要求“生成完全纯净”，而是通过 own-hit / foreign-hit 的黑盒指标判断隔离质量。

### 5.4 稳定性测试

稳定性分三层：

1. **新实例确定性**：相同 seed、相同写入顺序、相同 greedy prompt，是否给出完全相同结果
2. **同实例重复性**：同一实例上重复调用是否会自发漂移
3. **持久化精确性**：save/load 之后 greedy 输出是否保持完全一致

这三类一起能区分：

- 初始化不稳定
- 运行时状态漂移
- 持久化恢复偏差

## 6. 非 overfit 说明

第二轮仍然不要求“输出精确等于某句话”，因为那会把测试绑定到模型偶然文案。

本轮采用的稳定观测指标包括：

- prompt 前缀是否保留
- 是否成功扩展输出
- 是否出现目标领域关键词
- 是否出现非目标领域关键词
- greedy 输出是否在重复条件下完全一致

这类指标更稳健，也更符合黑盒验收目标。

## 7. 可执行资产

第二轮矩阵的可执行 runner：

- `/workspace/blackbox_test_agent_memory_round2.py`

支持两种运行模式：

### representative

执行每类 1 个代表场景：

- `R2-STRESS-01`
- `R2-LONG-01`
- `R2-CROSS-01`
- `R2-STABLE-01`

适合日常回归。

### full

执行全部矩阵场景：

- `R2-STRESS-01`
- `R2-STRESS-02`
- `R2-LONG-01`
- `R2-LONG-02`
- `R2-CROSS-01`
- `R2-CROSS-02`
- `R2-STABLE-01`
- `R2-STABLE-02`
- `R2-STABLE-03`

适合完整验证或发布前检查。

## 8. 推荐执行命令

代表集：

```bash
python3 /workspace/blackbox_test_agent_memory_round2.py --suite representative
```

全量集：

```bash
python3 /workspace/blackbox_test_agent_memory_round2.py --suite full
```

输出 JSON：

```bash
python3 /workspace/blackbox_test_agent_memory_round2.py \
  --suite representative \
  --json-out /workspace/reports/agent_memory_blackbox_round2_results.json
```
