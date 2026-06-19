# 稀疏训练行选择 — 全局跨数据集 indexer(training-set LOO)

生成时间: 2026-06-19
状态: ❌ **NO-GO(确凿)** — 扩规模(14 集)+ 满训练集(每集 240 query)全局训练后,
indexer 在 held-out 上 ~92% 的 cell 打不过 full;排除了"样本饥饿"这一最后疑点
脚本: `run_indexer_global.py`(4集)/ `run_indexer_scaleup.py`(14集)/ `run_indexer_fulltrain.py`(满训练集)
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

---

## 扩规模结果(14 数据集 × 2 seed = 28 cell,`run_indexer_scaleup.py`)

**WIN-RATE:indexer > full 仅 2/28(7%);indexer > knn 10/28(36%)。**

每数据集 `indexer−full`(2 seed 均值)**全部为负**(最好 −0.026,最差 pc1 −0.395)。
electricity 在 4 集单 seed 时的 0.821>full,扩规模后**转负(−0.039)** —— 确认是单点噪声。

## 满训练集结果(13 集 × 2 seed,每集 240 query = 3× 监督,`run_indexer_fulltrain.py`)

针对"之前只 80 query,是否样本饥饿"的质疑,把每数据集 query 提到 240(用满训练集做
leave-one-out,无泄露),focal_bce 向量化(400 epoch 28min→13s)。

**WIN-RATE:indexer > full 仅 2/26(8%);indexer > knn 8/26(31%)。**

唯一持平/微正:diabetes −0.002、steel-plates +0.000、pc1 +0.013。其余全负
(qsar −0.317、kc1 −0.276、ozone −0.228)。**3× 监督几乎没改变 win-rate(7%→8%)。**

## 最终结论:NO-GO,排除样本饥饿

1. **训练数据量不是瓶颈**:80 query(7%)、满训练集 240 query(8%)win-rate 几乎相同。
   增加监督有边际改善(phoneme 0.466→0.620→…),但收敛到一个 **< full 的天花板**。
2. **oracle 上限真实(Phase 0),但 learned indexer 即使用满训练集跨数据集训练,held-out 上
   ~92% cell 打不过 full**,连稳定超 KNN 都做不到(31%)。
3. **根因(全程一致)**:TabPFN 的行重要性是**集体性、行特定**的,不构成可跨 test 行/数据集
   泛化的"选行规则"——再多监督也学不出能迁移到未见行的选择函数。

这是 FlashMemory-on-TabPFN 方向最强的否定证据,实验设计干净(无泄露、满训练集、多 seed、
13–14 数据集、win-rate 而非挑数据集)。与整个研究一贯教训一致:**固定预训练 TabPFN 上,
推理时行选择/深度循环类技巧拿不到泛化的净增益。**
