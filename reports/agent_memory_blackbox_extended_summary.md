# AgentMemorySystem 扩展黑盒测试总览

## 1. 本轮完成内容

- 第二轮 full 覆盖
- 放大版跨域污染热图
- 第三轮 full 覆盖（边界输入、异常输入、性能/时延）

## 2. 第二轮 full 结果

- PASS: 7
- WARN: 2
- FAIL: 0

## 3. 第三轮 full 结果

- PASS: 10
- WARN: 1
- FAIL: 1

## 4. 放大版污染热图结论

- cooking: own=2, foreign=5, ratio=2.50, verdict=high-contamination
- finance: own=2, foreign=3, ratio=1.50, verdict=high-contamination
- music: own=8, foreign=2, ratio=0.25, verdict=mixed
- space: own=7, foreign=4, ratio=0.57, verdict=mixed

## 5. 最高优先级发现

- P1: 空 prompt `generate("")` 会直接崩溃。
- P1: `transformers 5.x` 兼容性失败仍然成立。
- P2: 跨域污染在 dual-domain、four-way 以及放大版热图中均被稳定复现。
- P3: `mt < 0` 缺少显式参数校验，当前表现为直接返回原 prompt。
