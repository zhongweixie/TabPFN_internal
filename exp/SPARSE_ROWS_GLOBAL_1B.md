# 稀疏训练行选择 — 全局跨数据集 indexer(training-set LOO)

生成时间: 2026-06-19
状态: ⚖️ 混合信号 — 全局训练部分挽回 Phase 1a 的样本饥饿失败,但未普适 GO
脚本: `reproduce/run_indexer_global.py`, `reproduce/indexer_global_run.out`
前置: `exp/SPARSE_ROWS_INDEXER_1A.md`(leak-free NO-GO,held-out 0.466)

---

## 动机

Phase 1a leak-free 失败(phoneme held-out AUC 0.466,比随机差)。怀疑根因之一是
**样本饥饿**:indexer 只在单数据集的 60 个 val query 上训练,不足以学到泛化的选行规则。
本实验用**跨数据集全局训练 + 更多 query** 检验:更多/更多样的合法监督能否抓住"行选择
一致性"并泛化到未见 test 行。

## 方法(无泄露)

每个数据集切三份(全部来自非 test 数据,互不相交):
- **context**(120):当 train context。
- **query**(80):做 **training-set LOO** 造 golden(query 行**不在** context 里,完全模拟
  test 行角色,但标签已知 —— 合法,因为不是真 held-out test)。`I[q,j]=p_full(y_q)−p_{-j}(y_q)`,
  全程在非 test 数据内。
- **test**(60):held-out,从不参与造 label,仅用于评估。

训练**一个** indexer 在汇集的所有 `(query_repr_d, ctx_repr_d, golden_d)` 上(跨数据集共享
参数 = 通用选行规则),held-out test 上评估。keep=25%,单 seed,4 数据集。

## 结果(held-out test,AUC)

| 数据集 | full | ceiling(oracle) | **indexer(held-out)** | KNN-local | 胜 full | 胜 KNN |
|---|---|---|---|---|---|---|
| breast_cancer | 0.989 | 1.000 | 0.913 | 0.967 | ✗ | ✗ |
| phoneme | 0.789 | 0.997 | 0.620 | 0.755 | ✗ | ✗ |
| **electricity** | 0.810 | 1.000 | **0.821** | 0.702 | ✓ | ✓ |
| qsar-biodeg | 0.860 | 1.000 | 0.808 | 0.806 | ✗ | ✓(微) |

## 结论(诚实)

1. **全局训练确实有效**:phoneme 从 Phase-1a 单数据集 leak-free 的 **0.466 → 0.620**
   (+0.154)。证明之前 NO-GO **部分是单数据集样本饥饿**,不全是"一致性不存在"。

2. **electricity 是 leak-free 下第一个真 GO 的 cell**:indexer 0.821 同时 > full(0.810)
   和 KNN(0.702)。说明跨数据集学到的选行规则**确实能泛化到该数据集的未见 test 行**。

3. **但不普适**:4 个里只有 1 个(electricity)真正胜 full;phoneme/qsar/breast_cancer 仍
   < full。"行选择一致性可跨数据集泛化"**部分成立,非普适**。

4. **修正了过早的 NO-GO**:结论从"learned indexer 不行"精确化为"**ceiling 真实 + 全局训练
   能在部分数据集让 indexer 超过 full,成功率取决于数据集 / 训练数据量**"。

## 边界

- 4 个小数据集、单 seed、N=120、layer-12 单层表示、keep=25%。
- electricity 的成功需多 seed 确认非单点幸运。
- query 行在自己 context 里、test 行不在 —— train/test query 的 distribution shift 仍在
  (但 TabPFN 本就如此工作)。

## 下一步

扩到 10+ 数据集 + 多 seed:检验 electricity 式成功是否随训练数据集数增多而变普遍
(跨数据集泛化通常随数据集数提升),还是始终零星。GO 判据:多数 held-out 数据集
稳健 > full + > KNN。
