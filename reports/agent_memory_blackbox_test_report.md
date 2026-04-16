# AgentMemorySystem 黑盒测试报告

## 1. 测试目标

对上传文件 `/home/ubuntu/.cursor/projects/workspace/uploads/AgentMemorySystem.md` 中的实现进行**黑盒测试**，要求满足：

- 不使用 mock
- 不简化被测逻辑
- 不修改被测代码
- 不写依赖内部实现细节的 overfit 测试
- 不依赖 fallback 路径掩盖真实问题

本次测试只通过其**公开可见运行行为**进行验证，不读取内部树结构、缓存、权重张量或私有辅助函数结果。

## 2. 被测对象

上传文档中包含完整可执行 Python 实现。黑盒测试仅使用以下公开调用方式：

- `MemLLM.load()`
- `MemLLM.write()`
- `MemLLM.generate()`
- `MemLLM.save_memory()`
- `MemLLM.load_memory()`

## 3. 测试环境

- OS: Linux 6.1.147
- Python: 3.12.3
- Torch: 2.11.0+cu130
- 模型: `gpt2`

本次测试先后验证了两套 `transformers` 环境：

1. `transformers 5.5.4`
2. `transformers 4.57.6`

## 4. 测试方法说明

### 4.1 黑盒边界

为了保持黑盒属性，测试中：

- 不调用被测文件自带的 `test()`、`test_*()` 内部测试函数
- 不读取 `amm.tree.store`、`_wte_neighbor_cache` 等内部状态
- 不通过 monkey patch、stub、fake model、替身 tokenizer 等方式替换真实依赖
- 不改源码、不降级功能、不删除逻辑

### 4.2 真实执行方式

测试采用真实依赖和真实模型执行：

- 真实加载 `gpt2`
- 真实调用 `write()` 写入记忆
- 真实调用 `generate()` 观察文本输出
- 真实保存/加载记忆文件

### 4.3 非 overfit 原则

断言不绑定某个固定完整句子，而只检查稳定且对外有意义的行为，例如：

- 是否成功加载
- 是否保留 prompt 前缀
- 是否在写入记忆后出现目标领域词
- 保存/加载后是否保留该领域响应能力

这避免了把测试写成“必须生成某一字不差文本”的脆弱用例。

## 5. 测试过程

### 步骤 A：定位公开入口

确认上传文档是完整可执行实现，并识别公开调用面：

- `load("gpt2")`
- `write(text, training_mode=True)`
- `generate(prompt, mt=..., greedy=True)`
- `save_memory(path)`
- `load_memory(path)`

### 步骤 B：环境准备

初始环境缺少 `torch` 和 `transformers`，先安装真实运行所需依赖。

### 步骤 C：兼容性复现

在 `transformers 5.5.4` 下直接进行真实调用，结果 `generate()` 崩溃，报错为：

`IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)`

崩溃栈位于 GPT-2 block 前向过程中，说明当前实现与 `transformers 5.x` 存在兼容性问题。

### 步骤 D：在兼容环境下执行正式黑盒用例

将 `transformers` 切换到 `4.57.6` 后重新执行同样的真实调用，功能恢复正常，然后运行独立黑盒测试驱动：

- 文件：`/workspace/blackbox_test_agent_memory_system.py`

## 6. 正式测试用例与结果

### TC-01 加载公开 API

**目标**  
验证 `MemLLM.load("gpt2")` 可在真实环境成功完成初始化。

**结果**  
通过。

---

### TC-02 空记忆下生成

**目标**  
验证未写入记忆时，`generate()` 可返回非空字符串，并保留输入 prompt 前缀。

**输入**  
`prompt = "Hello"`

**结果**  
通过。

**输出样例**

```text
'Hello the other a- I have in this, (the.\n "I'
```

---

### TC-03 写入前的音乐提示基线

**目标**  
记录未写入音乐记忆前，对音乐 prompt 的基线输出。

**输入**  
`prompt = "The piano performance"`

**结果**  
通过。

**输出样例**

```text
'The piano performance of the and, "The world (the-theon the other people in a. The on'
```

**观察**  
未出现命中的音乐领域关键词。

---

### TC-04 写入音乐记忆后观察领域接地

**目标**  
在写入真实音乐语料后，验证 `generate()` 是否出现可观察的音乐领域信号。

**写入内容**

1. `He practiced piano for hours perfecting a difficult Chopin nocturne.`
2. `She studied music theory and harmonic progression at the conservatory.`
3. `The orchestra rehearsed the symphony before the evening concert.`

**结果**  
通过。

**门控返回值**

- `0.552463`
- `0.654567`
- `0.569074`

**输出样例**

```text
'The piano performance musical music the and violin, a- is an in that\'s.\n The other " it has'
```

**命中关键词**

- `music`
- `musical`
- `violin`

**结论**  
从黑盒角度看，写入记忆后，生成结果出现了明确的音乐领域词，说明外部可观察的领域接地增强成立。

---

### TC-05 记忆前后领域信号增强

**目标**  
比较 TC-03 与 TC-04，确认写入记忆后领域信号相对增强。

**结果**  
通过。

**对比**

- 写入前关键词命中：`[]`
- 写入后关键词命中：`['music', 'musical', 'violin']`

**结论**  
在黑盒观察层面，写入记忆前后确实产生了显著可见差异。

---

### TC-06 记忆保存/加载回环

**目标**  
验证 `save_memory()` 与 `load_memory()` 后，模型仍保留可观察的音乐领域响应能力。

**结果**  
通过。

**中间文件**

- 大小：`25116 bytes`

**重载后输出样例**

```text
'The piano performance musical music the and violin, a- is an in that\'s.\n The other " it has'
```

**重载后关键词**

- `music`
- `musical`
- `violin`

**结论**  
从外部行为看，记忆持久化与恢复功能成立。

## 7. 汇总结果

在 `transformers 4.57.6` 环境下，正式黑盒测试结果：

- 通过：6
- 失败：0

总耗时约：

- `130.65s`

## 8. 发现的问题与风险

### P1：与 `transformers 5.x` 不兼容

**现象**  
在 `transformers 5.5.4` 环境中，公开接口 `generate()` 真实执行时直接崩溃。

**外部影响**  
这意味着如果用户在较新的 `transformers` 环境部署该实现，核心生成能力不可用。

**复现条件**

1. 安装 `torch 2.11.0+cu130`
2. 安装 `transformers 5.5.4`
3. 加载上传实现
4. 执行：

```python
m = MemLLM(Cfg())
m.load("gpt2")
m.generate("Hello", mt=15, greedy=True)
```

**结果**  
抛出：

```text
IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)
```

### P2：跨域污染风险

在额外探测中，如果同时写入音乐和太空两组记忆，`"The piano performance"` 与 `"The space telescope"` 两个 prompt 都可能混入另一领域词汇。

这说明该系统从黑盒现象上存在一定的**领域边界串扰**。  
本问题不影响本次主测试的“功能是否可用”结论，但会影响更高要求的语义隔离质量。

## 9. 结论

### 9.1 功能结论

在**兼容环境 `transformers 4.57.6`** 下，被测实现从黑盒角度表现为：

- 可以真实加载
- 可以在空记忆下生成
- 可以写入记忆并改变后续生成
- 可以在目标 prompt 上体现领域接地增强
- 可以保存和恢复记忆行为

### 9.2 质量结论

该实现具备可运行的外部功能闭环，但存在一个明确的工程风险：

- 对 `transformers 5.x` 的兼容性失败

因此，如果用于真实交付或部署，建议至少将运行环境版本要求显式固定，或后续再做兼容性修复验证。

## 10. 交付物

本次新增的测试资产：

- 黑盒测试驱动：`/workspace/blackbox_test_agent_memory_system.py`
- 测试报告：`/workspace/reports/agent_memory_blackbox_test_report.md`

## 11. 复现命令

在当前仓库根目录执行：

```bash
python3 /workspace/blackbox_test_agent_memory_system.py
```

如果要复现兼容性问题，可在 `transformers 5.x` 环境下执行真实调用进行验证。
