# Adaptive Loop (H5 ACT) 实验报告

更新时间：2026-07-11
状态：H5 训练、reference 对照评估与结果可视化均已完成。

## 1. 研究动机

Looped TabPFN 的固定深度 baseline 对每个任务使用相同的循环次数，例如 fixed
`k=4`。已有 loop 结果表明，更深的循环不是所有数据集上的统一最优选择：部分
任务从额外 refinement 中获益，部分任务则偏好较浅的循环，或从更深循环中获得的
收益很小。

因此本实验的问题是：

> 能否根据输入及中间隐状态，自适应地决定每个任务需要多少次 loop，而不是让全部
> 任务固定使用 `k=4`？

目标不是单纯让平均循环次数变小，而是在质量和计算之间建立更好的 Pareto trade-off：
简单任务少算，困难任务多算。

## 2. 当前 H5 ACT 设计

H5 使用 Adaptive Computation Time (ACT) 的 soft-halting 机制：

1. 最大 loop 深度为 `K_MAX=6`；
2. 每一轮根据 test-row hidden state 的均值预测 halt probability；
3. 所有循环层的 hidden states 按 halt probability 做可微加权组合；
4. 训练损失为：

```text
L = cross_entropy + λ × expected_ponder_cost
```

其中 `λ` 是计算惩罚系数。`λ` 越大，模型越被鼓励尽早停止；`λ` 越小，则更强调
保持预测质量。

本轮完成两组训练：

| Checkpoint | λ | 训练方式 |
|---|---:|---|
| `h5_act_l001_12L_80000` | 0.01 | 较弱的早停惩罚，优先质量-计算平衡 |
| `h5_act_l005_12L_80000` | 0.05 | 较强的早停惩罚，优先激进早停 |

两组均由 `curric_k1to6_12L_80000/step-60000.ckpt` 初始化，在 H800 Slurm
集群上使用 2 GPU、训练至 80K steps。

## 3. 最终评估协议

最终评估由 `reproduce/exp/looped_pretrain/run_h5_eval.py` 完成。

### 内部 TabArena-aligned protocol

- 数据：TabArena-v0.1 对应的 OpenML tasks；
- 范围：38 个、类别数不超过 10 的分类数据集；
- split：每个数据集使用 OpenML 官方 split 的 fold 0；
- 指标：macro ROC-AUC；
- 最大 train context：1,000 行；
- 最大 test rows：500 行；
- 最大特征数：40；
- 对照：独立训练的 fixed `loop-k=4` reference checkpoint。

这个协议用于 H5 variants 与 fixed-k baseline 的公平内部比较。它**不是**完整官方
TabArena leaderboard protocol，不能直接与 TabPFN-3 public leaderboard 数字比较。

完整 TabPFN-3 / TabArena protocol 使用所有 51 datasets、816 tasks（分类部分 38
datasets、681 tasks）、全部 splits、官方 task-specific metric、Elo/improvability
汇总，以及 bagging/refit 流程。

## 4. 最终结果

评估使用修复后的 reference 版本，覆盖 38 个数据集：

| 方法 | Mean AUC | Fixed-k=4 AUC | Mean expected k | 与 reference 的 AUC 差 |
|---|---:|---:|---:|---:|
| fixed `loop-k=4` reference | **0.8173** | — | 4.00 | 0.0000 |
| H5 ACT, λ=0.01 | 0.8157 | 0.7991 | 1.24 | -0.0016 |
| H5 ACT, λ=0.05 | 0.8087 | 0.6661 | 1.04 | -0.0086 |

### 结果解读

1. **λ=0.01 是当前最佳 H5 checkpoint。**
   ACT AUC 为 0.8157，只比 fixed `k=4` reference 低 0.0016；ACT 相比同一
   checkpoint 的 fixed-k=4 backbone 提升 0.0166，并在 38 个数据集中的 31 个上
   获胜。

2. **λ=0.05 更倾向于早停，但不够稳健。**
   平均 expected k 为 1.04，几乎总在第一步停止；其 ACT 输出仍达到 0.8087，
   但关闭 ACT 后的 fixed-k=4 backbone 只有 0.6661，显示较强的 backbone 退化。

3. **当前 adaptive loop 尚未超过 fixed `k=4`。**
   最佳 adaptive 结果接近但没有超过 fixed-k reference。当前结论应表述为：

   > Soft ACT 基本保持了 fixed-k=4 的预测质量，并学习出偏向较浅循环的输出权重；
   > 但尚未证明 adaptive depth 在质量上优于 fixed depth。

4. **当前 mean k 不等于真实 runtime 加速。**
   现有 soft ACT forward 会先执行完整的 `K_MAX=6` 次循环，再对各层输出加权。
   因此 `mean k=1.24` 是 learned expected depth / ponder cost，而不是实际 GPU
   只执行 1.24 次循环。当前没有获得真实 wall-clock 推理加速。

## 5. 已修复的 evaluator 问题

Reference checkpoint 最初缺失，补充下载后发现：

```text
build_model(reference) 会重新安装全局 fixed-loop LayerStack.forward，
从而覆盖 H5 的 ACT forward。
```

这会导致错误评估：ACT `mean k` 变成 6.00，实际运行的是 fixed loop。

修复方式是在加载 reference model 后重新调用 `install_act_forward()`：

```python
from train_h5_act import install_act_forward
install_act_forward()
```

修复后的结果与无 reference 时的 ACT AUC / mean-k 一致，因此本报告只采用
`*_with_ref_fixed.json` 的结果。

## 6. 可迁移的重要文件

### 代码

- `reproduce/exp/looped_pretrain/train_h5_act.py`：H5 ACT 训练实现；
- `reproduce/exp/looped_pretrain/train_h6_budget_cond.py`：H6 Budget-Conditioned 对照；
- `reproduce/exp/looped_pretrain/run_h5_eval.py`：H5 最终评估（包含本轮 reference
  forward 修复）；
- `reproduce/exp/looped_pretrain/run_h6_eval.py`：H6 最终评估；
- `reproduce/exp/looped_pretrain/looped_step2.py`：loop / re-injection 基础实现；
- `reproduce/exp/looped_pretrain/slurm_h5_l001.sh`、`slurm_h5_l005.sh`、
  `slurm_h6.sh`：Slurm 训练脚本（仓库已有版本）。

### 结构化结果与可视化

- `reproduce/h5_l001_eval_with_ref_fixed.json`；
- `reproduce/h5_l005_eval_with_ref_fixed.json`；
- `reproduce/h5_final_results.html`；
- `reproduce/ADAPTIVE_LOOP_REPORT.md`（本报告）。

### Checkpoints（不进入 Git）

- `reproduce/ckpt/h5_act_l001_12L_80000/step-80000.ckpt`；
- `reproduce/ckpt/h5_act_l005_12L_80000/step-80000.ckpt`；
- `reproduce/ckpt/loopk4_12L_80000/step-80000.ckpt`；
- base checkpoint：GitHub Release `v1.0-base-ckpt`。

## 7. 下一步建议

### 第一优先级：Oracle Depth 诊断

在同一个 fixed backbone 上系统评估 `k=1...6`，记录每个数据集/episode 的最佳 k：

```text
oracle_k = argmin_k (prediction_loss_k + compute_cost × k)
```

需要回答：

- 不同数据集的最佳 k 是否稳定且不同？
- 一个知道最优 k 的 oracle，相比 fixed `k=4` 的上限增益是多少？
- 当前 halt policy 的 predicted k 是否与 oracle k 有相关性？

如果 oracle gain 很小，adaptive loop 的质量上限有限；重点应转向真实加速。
如果 oracle gain 显著但 halt policy 学不到，则说明 policy/训练机制需要改进。

### 第二优先级：Value-based / Oracle-supervised policy

当前 ponder penalty 只提供“整体少算”的压力，没有直接训练：

```text
从 k 到 k+1 是否能带来足够的预测收益？
```

可在训练 episode 中使用 held-out labels 构造：

```text
gain_k = loss_k - loss_(k+1)
```

训练 policy 预测 next-step value，只有 predicted gain 大于计算成本时才继续。

### 第三优先级：Multi-exit + teacher distillation

让每个 k 都产生可用预测，并用 fixed-k=4 teacher 对较浅 exit 做蒸馏。这样停止策略
不再依赖未充分训练的浅层 representation。

### 第四优先级：Hard early exit

部署时必须按以下方式真正停止：

```text
for k in 1...K_MAX:
    run one loop
    if policy says stop:
        return current prediction
```

之后报告真实 latency、throughput、AUC 与平均实际执行 k 的 Pareto 曲线。

### 可替代主线：H6 + 独立 selector

H6 将“给定 budget k 时如何预测”和“选择哪个 k”解耦。可以先用 H6 建立
不同 fixed-k 的 quality curve，再使用 oracle best-k 标签训练独立 selector。这比
H5 同时优化 backbone、halt head 和 ponder loss 更容易诊断，也更不容易损伤 backbone。
