# FlashMemory-on-TabPFN: 学习式稀疏训练行选择 — 实现计划

生成时间: 2026-06-18
状态: 规划完成,待执行阶段 0

## 背景与定位(经多轮审查后确立)

把 FlashMemory-DeepSeek-V4 的"冻结主干 + 解耦训练轻量索引器"范式接到 TabPFN v2.5:
训练一个轻量 indexer,为每个 test 行预测它该 attend 哪些训练行子集。目标:大 N 下省算力 + 去噪。

定位(见 memory: tabpfn-sparse-row-selection-direction):
- per-query 白盒注意力路线,是 VIP-COP 显式放弃的赛道(它做 global context、黑盒 SHAP)。
- 真正要打败的基线 = per-query KNN local context(LoCalPFN 式),不是 random。

## 之前 training-free 探针失败的教训(不重复)

- training-free 阈值化 = H2O/Quest 式启发式,正是 FlashMemory 要超越的对象,不能回答
  "learned 行不行"。
- 我之前"实体化主注意力权重做选择" → electricity OOM(27GB)。FlashMemory 用轻量低秩
  indexer,不碰主注意力权重,才省显存。本计划照 FlashMemory 做,绕开 OOM。

## 关键可行性(已探索确认,附 file:line)

1. **LOO 高效可行**:KV cache 是 per-row `(B,N,1,head_dim)`(kv_cache.py:60),删一行 = cat
   切片(tabpfn_v2_5.py:266-276 cache 路径),无需重算训练编码 → LOO ≈ N 次廉价 test 注意力。
2. **Indexer 挂载点**:24 层中的 mid-late(如 11/14/18);用 pre-projection `x_BcRE`
   (tabpfn_v2_5.py:263)作冻结 key 表示;新建低秩双塔投影(原 q/k/v 全秩 192→192,无现成
   低秩);thinking rows(前 64)不计分,只对真训练行打分。
3. **TabArena 复用**:run_full_tabarena.py 有 51 数据集 + 加载 + ROC-AUC;缺 per-fold min-max
   归一化(TabPFN-3 §F.1)+ 多 seed,需补。

---

## 阶段 0:LOO 真上限闸门(go/no-go,先做)

回答唯一决定性问题:**存不存在一个训练行子集能保住/超过 full?**(之前的假 oracle 不算数)

- KV-cache 切片高效 LOO:小 test 集 + 中等 N(~2000),算每个训练行对每个 test 行的影响
  `I_j = p_full − p_{-j}`。
- 取 LOO-top-k% 行作为 per-query 子集,测 ROC-AUC vs full,扫 k∈{10,25,50,75,90}%。
- 干净数据 + label-flip 难噪声(翻 50% 真实行标签;已实测 full 0.92→0.29)两种设定都测。
- 判据:干净打不过 full 且噪声不能恢复 → no-go;噪声下能滤掉翻转行、恢复精度 → ceiling
  存在,进阶段 1。

脚本:`reproduce/run_loo_ceiling.py`,报告 `exp/SPARSE_ROWS_LOO_CEILING.md`。

---

## 阶段 1:学习式 indexer(若 ceiling 存在)— 分两 branch

照 FlashMemory indexer(低秩双塔 + sigmoid + focal loss γ=2 + 3:1 负采样 + decoupled,
冻结 TabPFN,只训新投影矩阵)。**两 branch 唯一区别 = golden label 来源**(分叉点):

### Branch `idx/loo`:golden = LOO 影响(行为定义,真但贵)
golden 行 = LOO 影响 top-k。最忠实"重要性",label 造价高(仅小集合)。

### Branch `idx/voting`:golden = FlashMemory 跨层注意力投票(便宜近似)
照 FlashMemory §2.2:每层 softmax → top-p(0.6)二值化 → 跨层计数 ≥θ(3)= golden。
(是"二值化+投票",不是之前 oracle 的"跨层相加",后者审查已证不稳健。)label 只需一次
full forward,便宜可规模化。

对比:哪种 label 训出的 indexer 更强、是否都 > KNN。直接回答"贵的行为定义值不值,
还是便宜的注意力投票够用"。

indexer 结构(两 branch 共享):test 行 mid-late 层 pre-projection → 低秩 query(192→r,
r≈64);训练行冻结 key 表示同层 → key 投影(192→r);score = sigmoid(head-fused q·k);
阈值/top-k 选子集;focal loss + 3:1 负采样;冻结全部 TabPFN,只训 indexer 投影。

脚本:`reproduce/run_indexer_train.py`(参数化 label 来源)。

---

## 阶段 2:评估(对齐 TabPFN-3 论文)

- benchmark:TabArena 子集(复用 run_full_tabarena.py)+ large-data 大 N regime。
- 指标:ROC-AUC(OvR 多类)+ TabPFN-3 §F.1 per-(dataset,fold) min-max 归一化(需新写)。
- 对照:full / indexer-loo / indexer-voting / KNN-local / random,同 keep 预算。
- 统计:多 seed + Friedman/Conover CD 图。
- 去噪:label-flip 与/或 VIP-COP DN(TabPFN-3 自身不测噪声)。

---

## 执行顺序

1. 阶段 0 LOO 上限闸门(干净 + 噪声)→ 决定 go/no-go。← 先做
2. 若 GO:从 main 拉 `idx/loo`、`idx/voting`,各实现 + 训练 + 评估。
3. 阶段 2 规范评估 + 两 branch 对比 + 对 KNN。
4. 汇总报告。

所有改动在 fork 的 main/分支,默认推理零影响(沿用 thinking-mode 运行时旁路约定)。
