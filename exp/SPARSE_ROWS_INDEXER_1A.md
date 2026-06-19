# 稀疏训练行选择 — 阶段 1a:学习式 indexer 机制闸门报告

生成时间: 2026-06-19
状态: ❌ **NO-GO** — leak-free 重测确认:学习式 indexer 不泛化(phoneme held-out AUC 0.466)
脚本: `reproduce/run_indexer_gate.py`(`run_dataset_honest` = leak-free 协议)
前置: 阶段 0 `exp/SPARSE_ROWS_LOO_CEILING.md`(证明上限存在)

---

> ## ⚠️ 重大更正(2026-06-19,审查后)
>
> 本报告原结论"GO,indexer 达上限 94%(AUC 0.987)"**是数据泄露造成的假象,已撤销。**
>
> **泄露链**:idx/loo 的 golden label 由 `loo_influence(...,yte)` 算出 —— 用了 **test 标签**
> (每个 test 行真标签下的 LOO 影响)。indexer 训练目标 = 这些含 test 答案的 label,然后在
> **同一批 test 行**上评估(transductive)。indexer 实质是"记住了这批 test 行该选哪些行",
> 不是学到泛化选行规则。
>
> **决定性实测**(phoneme,golden label 用 test 集 A 造、在不相交 held-out 集 B 评估):
>
> | 评估集 | full | indexer |
> |---|---|---|
> | A(造 label 用的集,泄露) | 0.873 | **0.960** |
> | **B(held-out,诚实)** | 0.849 | **0.688** |
>
> 同一个 indexer:泄露集 0.960,诚实集 **0.688 —— 远低于 full(0.849)和 KNN(0.814)**。
> 泄露虚高了 ~0.27 AUC。**当前实现的 indexer 不会泛化到未见过的 test 行。**
>
> **正确协议(已执行,见下)**:golden label 在 **held-out validation 集**上算(用 val 标签),
> indexer 在从未参与造 label 的 test 集上评估。`run_dataset_honest` 实现此协议。

---

## Leak-free 最终结果(`run_dataset_honest`,train/val/test 三分,val 造 label、test 评估)

split:train 120(context)/ val 60(LOO 造 label)/ test 60(评估,从不参与造 label)。
indexer 在 val 行上学"选哪些训练行",应用到**未见 test 行**。keep=25%,单 seed。

| 数据集/branch | ceiling(oracle,test) | full | **indexer(held-out)** | KNN-local | 胜 full | 胜 KNN |
|---|---|---|---|---|---|---|
| breast_cancer/loo | 1.000 | 0.989 | 0.875 | 0.967 | ✗ | ✗ |
| breast_cancer/voting | 1.000 | 0.989 | 0.980 | 0.967 | ✗ | ✓(微) |
| **phoneme/loo** | 0.997 | 0.789 | **0.466** | 0.755 | ✗ | ✗ |
| phoneme/voting | 0.997 | 0.789 | 0.615 | 0.755 | ✗ | ✗ |

**裁决:NO-GO。** 没有一个 cell 的 indexer 超过 full。phoneme/loo 从泄露版 0.987 崩到
**0.466(比随机还差)** —— 决定性证明 0.987 几乎全是 test 标签泄露。

## 结论(诚实)

1. **上限是真的**(ceiling 0.997–1.000,阶段 0 已证:对每个特定 test 行存在好子集)。
2. **但学习式 indexer 抓不住它泛化到未见 test 行**:indexer 学的"选行规则"过拟合见过的
   val 行,迁移到 test 行就崩(phoneme held-out 0.466)。
3. **idx/loo 在 leak-free 下反而比 idx/voting 差**(phoneme 0.466 vs 0.615):LOO label 最依赖
   具体行的 test 标签,泛化性最差;voting 基于注意力(与标签无关)迁移略好。这个相对泄露版的
   **反转本身印证了泄露的存在**。
4. **根本障碍**:TabPFN 对训练行的使用是集体性、行特定的——"哪些行重要"**不构成可跨 test 行
   泛化的规则**。oracle 上限对每个特定行存在,但无法从其他行预测出来,故 learned indexer 此路不通。

这与整个研究的一贯教训一致(thinking-mode、training-free 探针):**在固定预训练 TabPFN 上,
推理时的行选择/深度循环类技巧难以拿到泛化的净增益。**

## 方法论价值(这次研究链的正面收获)

负结果本身可信,且过程演示了严谨性:training-free 探针 bug → LOO 上限证明 → indexer 实现两
bug(hook 漏 thinking 行、relu 死区)修复 → **发现并实测确认 test 标签泄露** → leak-free 重测
得到真实 NO-GO。每一步"漂亮结果"都经住了或倒在了对抗性检验下。

---

## 目的

阶段 0 证明了"存在 per-query 训练行子集能超过 full"(oracle 上限,用了 test 标签)。
阶段 1a 问:**一个不看 test 标签的轻量学习式 indexer,能否逼近这个上限、且超过 KNN?**
两 branch 只换 golden label 来源(idx/loo vs idx/voting)。

## 方法(FlashMemory 式,decoupled)

- **indexer**:低秩多头双塔(LayerNorm 输入 + 无 relu 双线性 + head routing),~78k 参数,
  冻结 TabPFN,只训 indexer 投影。输入 = layer-12 跨列 mean-pool 的 per-row 表示。
- **golden label** [M,N]∈{0,1},两 branch:
  - loo:LOO 影响 top-25%(复用阶段 0 `loo_influence`)。
  - voting:跨层注意力 top-p(0.1)二值化 + 投票 ≥3(p=0.6 原值在小 N 退化成全选,已调)。
- **训练**:focal loss + 3:1 负采样,400 epoch,decoupled(backbone 不在环)。
- **推理 hook**:gather-then-SDPA,test 行 attend thinking 行 + indexer 选的 top-25% 训练行
  (不实体化 (M,H,N) 权重)。
- 设置:N=120, M=40, 单 seed, keep=25%。

## 修复的两个关键 bug(否则结论是假的)

诊断阶段(多 subagent + 亲自实测)发现首版结果(indexer 全面劣于 full、两 branch 完全相同)
是两个实现 bug 造成的**假 NO-GO**:
1. **推理 hook 漏掉 thinking 行**:test 行只 attend 选中训练行,排除了 64 个 thinking 行 →
   keep=100% 都不等于 full(maxdiff 0.34)。修:候选 = thinking 行 + 选中训练行 → maxdiff 0.0016。
2. **indexer relu 死区**:真实 layer-12 表示(norm~12,未归一化)喂入 relu 双塔 → logit 塌成
   常数(std→0)、PR-AUC=chance(0.25)。修:去 relu + 输入 LayerNorm → PR-AUC 0.25→0.66。
   (注:subagent 曾误判根因为"focal_bce 随机负采样",经实测证伪——随机数据上 PR-AUC 0.95。)

## 结果(修复后)

| 数据集 | branch | full auc | ceiling(LOO) | **indexer** | KNN-local | 达上限% | 胜KNN |
|---|---|---|---|---|---|---|---|
| breast_cancer | loo | 0.972 | 1.000 | 0.934 | 0.941 | −136% | ✗ |
| breast_cancer | voting | 0.972 | 1.000 | 0.959 | 0.941 | −45% | ✓ |
| **phoneme** | **loo** | 0.848 | 0.996 | **0.987** | 0.814 | **94%** | ✓ |
| phoneme | voting | 0.848 | 0.996 | 0.853 | 0.814 | 3% | ✓ |

## 结论

1. **GO 信号(phoneme/loo)**:不看 test 标签的学习式 indexer 达 auc 0.987 = oracle 上限的
   **94%**,远超 full(0.848)和 KNN(0.814)。**FlashMemory 式范式在 TabPFN 上首次拿到正面、
   可信的机制证据。**

2. **idx/loo > idx/voting**(branch 对比的答案):phoneme 上 loo(0.987)远胜 voting(0.853)。
   **LOO 行为定义的 golden label 值这个贵** —— voting 的便宜注意力近似训出的 indexer 弱。

3. **难数据集才显价值**:breast_cancer 太易(full 0.97),indexer 打不过 full;phoneme 难
   (full 0.85)才显出 indexer 大幅超越 full+KNN。

4. **两 bug 修复彻底翻转结论**:修复前 phoneme/loo indexer=0.632(< KNN,灾难);修复后
   =0.987。又一次"修对实现,结论翻转"——印证了诊断而非急于下结论的价值。

## 诚实的边界

- N=120、单 seed、2 数据集 —— **最小机制闸门,不是性能证明**。phoneme/loo 强,但需多 seed/
  多数据集确认稳健,不能据此声称普适。
- per-dataset / transductive(indexer 用该数据集自己的 LOO label 训,在同一 test 集评估)。
  **cross-dataset 迁移(1b)未测** —— 那才是 TabPFN 式零样本价值。
- 噪声设定(label-flip)下 indexer 表现未测(阶段 0 证明上限在噪声下尤其高)。

## 下一步

1. **扩规模**:多 seed + 更多数据集,确认 phoneme/loo 不是单点幸运;加噪声设定。
2. **1b cross-dataset**:在若干数据集训 indexer、held-out 测,验证迁移(若 1a 稳健)。
