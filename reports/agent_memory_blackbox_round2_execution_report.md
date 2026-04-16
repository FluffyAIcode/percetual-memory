# AgentMemorySystem 第二轮黑盒测试执行报告

## 1. 执行范围

本次执行基于第二轮矩阵 runner：

- `/workspace/blackbox_test_agent_memory_round2.py`

执行套件：

- `representative`

代表集覆盖四类风险面各 1 个场景：

1. 压力测试
2. 长文本测试
3. 跨域污染测试
4. 稳定性测试

## 2. 执行环境

- OS: Linux 6.1.147
- Python: 3.12.3
- Torch: 2.11.0+cu130
- Transformers: 4.57.6
- Base model: `gpt2`

## 3. 执行命令

```bash
python3 /workspace/blackbox_test_agent_memory_round2.py \
  --suite representative \
  --json-out /workspace/reports/agent_memory_blackbox_round2_results.json
```

## 4. 总体结果

- PASS: 3
- WARN: 1
- FAIL: 0
- 总耗时: `321.42s`

结论：

- 第二轮代表集在兼容环境下整体可执行
- 没有出现新的阻断性故障
- 跨域污染问题被稳定复现并分类为 `WARN`

## 5. 分场景结果

### R2-STRESS-01 repeated write/generate pressure

**类型**：压力测试  
**结果**：PASS

**结论**

- 在两轮连续写入四个领域语料、并对四个 prompt 连续生成的压力下，没有出现崩溃
- `write()` 返回 gate 值有限
- `generate()` 能持续返回有效字符串，并保留 prompt 前缀

**关键指标**

- rounds: `2`
- total_writes: `24`
- total_generations: `8`
- avg_gate: `0.564429`

**样例输出**

- music:
  `The piano performance musical music the mission of increased, and reduce a- or in this reduced to be more that is`
- space:
  `The space telescope planets around musicals mission- the and increased. reduce team of a, that's\n in this`

**观察**

功能层面通过，但混合域压力下输出中已能看到跨域词汇渗透，这与后续污染场景结果一致。

---

### R2-LONG-01 long memory write grounding

**类型**：长文本测试  
**结果**：PASS

**结论**

- 单条超长记忆文本写入后，系统仍能对目标 prompt 产生可见的音乐领域接地

**关键指标**

- long_text_chars: `4099`
- gate: `0.293301`
- keyword_hits:
  - `concert`
  - `music`
  - `musical`
  - `piano`
  - `practice`
  - `theory`

**输出样例**

```text
The piano performance piano Music Practice music practice musical theory concerting hard, night and hours of the morning hour
 a- in this evening The
```

**观察**

长文本写入并未导致接口失效，说明至少在这一级别的长输入下，系统仍保持外部可用性。

---

### R2-CROSS-01 dual-domain contamination diagnostic

**类型**：跨域污染测试  
**结果**：WARN

**结论**

- 双域混合写入后，音乐 prompt 与太空 prompt 都出现明显的跨域串扰
- 该问题不是崩溃类问题，但会影响语义隔离质量

**music prompt 结果**

- own hits:
  - `music`
  - `musical`
- foreign hits:
  - `mission`
  - `planet`

**space prompt 结果**

- own hits:
  - `mission`
  - `planet`
- foreign hits:
  - `music`
  - `musical`
  - `theory`

**样例输出**

- music:
  `The piano performance musical music the mission of a, and planets-\n. in that is an all other's to`
- space:
  `The space telescope planets beyond the musical theory of a- and mission\n, in that is an all other. to`

**判定原因**

该场景是 diagnostic 场景，不把污染直接判成 FAIL；但由于 own-domain 与 foreign-domain 词同时明显出现，因此记为 `WARN`。

---

### R2-STABLE-01 fresh instance determinism

**类型**：稳定性测试  
**结果**：PASS

**结论**

- 在相同 seed、相同写入顺序、相同 greedy prompt 下，全新实例之间的输出完全一致

**输出**

```text
The piano performance musical music the and violin, a- is an in that's.
 The other " it has
```

**意义**

这说明在当前兼容环境中，初始化路径和 greedy 解码路径具备可重复性。

## 6. 结果解读

### 6.1 压力层面

第二轮代表集没有发现新的崩溃型问题。  
在较高调用密度下，公开接口仍然可用。

### 6.2 长文本层面

超长记忆写入场景通过，说明记忆写入链路对较长输入具备一定耐受性。

### 6.3 语义隔离层面

跨域污染问题被再次稳定复现，是本轮最重要的质量风险。  
它不会阻止系统“运行”，但会削弱“提示词所属领域 -> 目标领域输出”的纯度。

### 6.4 稳定性层面

新实例确定性通过，说明在兼容环境和 greedy 模式下，结果具备良好的可复现性。

## 7. 后续建议

如果继续做第三轮，可以优先补这几项：

1. `R2-STRESS-02 repeated save/load pressure` 的全量执行
2. `R2-LONG-02 long prompt resilience` 的边界行为验证
3. `R2-CROSS-02 four-domain contamination matrix` 的全量污染热图
4. `R2-STABLE-02` 与 `R2-STABLE-03` 的完整执行

## 8. 关联文件

- 第二轮矩阵设计：
  `/workspace/reports/agent_memory_blackbox_round2_matrix.md`
- 第二轮代表集结果 JSON：
  `/workspace/reports/agent_memory_blackbox_round2_results.json`
- 第二轮执行报告：
  `/workspace/reports/agent_memory_blackbox_round2_execution_report.md`
