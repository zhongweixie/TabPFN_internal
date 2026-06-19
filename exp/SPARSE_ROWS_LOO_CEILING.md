# 稀疏训练行选择 — 阶段 0:LOO 上限闸门报告

生成时间: 2026-06-18
状态: ✅ 完成 — **GO**(ceiling 强烈存在,进阶段 1)
脚本: `reproduce/run_loo_ceiling.py`, `reproduce/loo_ceiling_run.out`
计划: `exp/SPARSE_ROWS_PLAN.md`

---

## 目的

回答方向的 go/no-go 闸门:**存不存在一个 per-query 训练行子集能保住/超过 full?**
之前的探针(v1/v2)用模型自己的注意力当重要性(启发式),且有 head-0 bug / 弱噪声 /
假 oracle 等缺陷,不能定论。这里改用**留一法影响(LOO influence)**——行为层面的真值。

## 方法

- **真 LOO(非 cache 切片)**:删一行就重训 `clf.fit(X\{j})`。原因:训练行的 cached key 是
  attend 全体训练行 + 24 层混合的产物,删 cache 某行 ≠ 那行不存在(实测 cache-slice vs
  refit maxdiff 0.37,不等价)。故只能朴素重训,限小 N。
- N_train=120, N_test=40, 2 seeds, 单 estimator。
- 影响 `I[m,j] = p_full(true_m) − p_{-j}(true_m)`(正 = 行 j 对 test m 有帮助)。
- 每个 test 行取其 top-k% 影响行作子集,**只用该子集预测该行**,扫 k∈{10,25,50,75,90}%。
- 两设定:clean + **label-flip 50%**(翻转 50% 训练行标签,on-manifold 难噪声)。
- 指标:accuracy + ROC-AUC。

## 结果(Δ 相对 full;2 seeds mean±std)

| 数据集 | 设定 | full acc / auc | LOO 最优 acc / auc | Δacc 最优 |
|---|---|---|---|---|
| breast_cancer | clean | 0.950 / 0.987 | **1.000 / 1.000** (k=10%) | +0.05 |
| breast_cancer | flip50 | 0.587 / 0.555 | **1.000 / 1.000** (k=25–75%) | +0.41 |
| phoneme | clean | 0.800 / 0.900 | **0.988 / 1.000** (k=50%) | +0.19 |
| phoneme | flip50 | 0.425 / 0.445 | **0.988 / 1.000** (k=25%) | +0.56 |
| electricity | clean | 0.788 / 0.850 | **0.988 / 1.000** (k=25–50%) | +0.20 |
| electricity | flip50 | 0.438 / 0.440 | **1.000 / 1.000** (k=25%) | +0.56 |

`frac_neg_influence ≈ 0.43–0.55`(约半数训练行影响为负,即"该删")。

## 结论

1. **Ceiling 存在(clean)**:per-query LOO 子集只用 25–50% 行就稳定超过 full(+0.05~+0.20,
   std 小)。推翻之前几轮"重要性不集中→无 headroom"的悲观——前面没看到它是因为用的是
   注意力启发式 / 有 bug,不是真影响。

2. **去噪假设成立(噪声,翻盘的一击)**:label-flip 50% 后 full 崩到 0.43–0.59;LOO 选
   top-25% 行直接恢复到 0.99–1.00(+0.41~+0.56)。这是 FlashMemory "less is more" 在 TabPFN
   的体现 —— **存在子集能滤掉被污染行、把模型从崩溃救回满分**。这正是 v1/v2 因弱噪声 +
   假 oracle 没测到的场景。

3. **frac_neg≈0.5 与结论自洽**:约半数行有害,恰好解释"为何选择有用",不矛盾。

## 必须诚实标注的边界

- **这是 ORACLE 上限**:LOO 影响用了 **test 标签**选行,不是可部署性能。它回答"最好能多好"
  (天花板很高)。**indexer 阶段才要在不看 test 标签下逼近它** —— 这是 go/no-go 闸门,不是
  最终性能声明。
- 小 N(120)、2 seed、3 数据集、单 estimator。上限探针,非规范评估。
- 噪声=label-flip(标签污染)。其他噪声类型(特征噪声、增广无关行)未测。

## 裁决:GO → 阶段 1

闸门通过:ceiling 又高又稳,噪声下尤其强。投入训练 FlashMemory 式轻量 indexer 有据可依
(目标 = 逼近此 LOO 上限,且不看 test 标签)。

下一步(阶段 1,分两 branch,见 [[SPARSE_ROWS_PLAN]]):
- `idx/loo`:golden label = LOO 影响 top-k(真但贵)
- `idx/voting`:golden label = FlashMemory 跨层注意力 top-p 投票(便宜近似)
- 同结构、同评估,只换 label;对比两者 + 对 KNN。
