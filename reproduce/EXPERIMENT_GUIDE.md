# 实验背景与后续指导：Adaptive Halting for Looped TabPFN

## 1. 背景：TabPFN 与 Looped Predictor

**TabPFN** 是一个用于小样本表格数据分类的 in-context learning 模型（Transformer）。
它直接把训练集 (X_train, y_train) 和测试集 X_test 一起喂进去，输出每个测试样本的类别概率——不需要梯度更新，推理即预测。

**Looped Predictor** 是在 TabPFN 基础上的改进：把 Transformer encoder（LayerStack）循环跑 K 次，每次重注入原始 token：

```
h = x                         # x = concat(train tokens, test tokens)
for step in range(K):
    h = LayerStack(h)
    h = h + alpha * h0        # h0 = 原始 token（固定）
output = head(h[test_rows])
```

K 越大，"思考"越深，精度越高——但计算量也线性增加。
**问题**：能不能对"简单样本"用小 K，"难样本"用大 K，在精度不损失的前提下节省计算？

---

## 2. 研究目标

**核心问题**：为 Looped TabPFN 设计一个 adaptive halting 机制，使平均循环次数 mean_k < 4（固定 k=4 的基准），同时 TabArena AUC 不下降（基准 ≈ 0.810）。

**理论上界**：用 curric-60K checkpoint 在 eval_bank 上，对每个数据集独立选最优 k（oracle），AUC ≈ 0.893。说明"选对 k"有约8个点的收益空间。

---

## 3. 实验方案演进

### 3.1 已关闭方向（H1–H3）

| 方案 | 思路 | 结果 | 失败原因 |
|------|------|------|---------|
| **H1** 置信度阈值 | 当前步输出最大 softmax 值超过阈值就停 | AUC 0.7981 | 置信度不等于正确率，容易过早停在错误输出 |
| **H2** KL稳定性 | 相邻两步输出分布 KL < ε 就停 | AUC 0.7974 | 分布收敛不等于分布正确，同样过早停 |
| **H3** Halt MLP | 训练一个小 MLP，输入当前隐状态，预测"现在停止是否准确" | AUC 0.7867，mean_k=1.00 | Oracle 设计缺陷：合成数据精度始终 >0.5，halt head 无差别预测高值，全部 k=1 停 |

**结论**：H1/H2 是 post-hoc（不改变模型），信号弱。H3 oracle 设计有根本缺陷（数据集级 accuracy 不适合做行级 halt 标签）。这三个方向全部关闭。

### 3.2 在训方向（H5/H6）

| 方案 | 思路 | 当前状态 |
|------|------|---------|
| **H5 ACT** | Adaptive Computation Time（Graves 2016）。每步维护 halt 概率 p_k，输出是所有步的加权和，ponder cost 作为正则 | 本机已训到 step 67K（单卡 L20），acc ≈ 0.63+ |
| **H6 Budget-Cond** | 训练时随机采样目标 budget k，把 budget_emb[k] 注入 loop 输入。测试时用 H1-style 阈值选 k | 本机同上 |

---

## 4. 关键实现细节

### H5 ACT（`train_h5_act.py`）

**ACT 权重计算**（Graves 2016，修复版）：
```python
weights = []
remaining = ones(B)
for p in halt_probs:         # p[k] = sigmoid(proj(h_mean_k))
    w = remaining * p        # 所有步都参与，包括最后一步（避免 dead gradient）
    remaining = remaining * (1 - p)
    weights.append(w)
# 残差保证 sum(weights) == 1
weighted_out += remaining * hiddens[-1]
ponder += k_max * remaining   # 未用完的 budget 算最大代价
```

**Ponder loss**：`total_loss = ce_loss + lambda * mean(ponder_cost)`

**两个超参数变体**：
- `h5_act_l001`：`ACT_LAMBDA=0.01`（弱惩罚，偏高精度）
- `h5_act_l005`：`ACT_LAMBDA=0.05`（强惩罚，偏少步数）

**eval 时**（`run_h5_eval.py`）调用 `enable_act(n_train=n_train)` 激活 ACT 模式，`_ACT_STATE["ponder_cost"]` 读平均步数。

### H6 Budget-Cond（`train_h6_budget_cond.py`）

**关键注入点**：
```python
# 1. 在 loop 开始之前注入 budget（k=1 时也能得到信号）
k_emb = budget_emb(k_tensor)   # Embedding(k_max+1, emsize)
h = h + broadcast(k_emb)
h0 = x                          # h0 固定为原始 token，不受 budget 影响
# 2. 每步 re-injection 时注入剩余 budget
rem_emb = remaining_emb(remaining_tensor)
h = h + alpha * h0 + broadcast(rem_emb)
```

**DDP 同步**（post-optimizer weight averaging，非梯度同步）：
```python
# run_batch() 里 super().run_batch() 调完之后
for p in self.raw_model.budget_embedding.parameters():
    dist.all_reduce(p.data, op=dist.ReduceOp.AVG)
```
原因：`budget_embedding` 不参与 standard DDP gradient sync（在 `super().__init__` 外注册），post-optimizer averaging 数学等价于 gradient sync。

**eval 时**（`run_h6_eval.py`）每次 predict 必须：
```python
enable_budget(k=k)
proba = model(X_test, y_train)
disable_budget()
```

### 通用注意事项

- **模型规格**：emsize=192，12 layers，max_classes=10
- **base checkpoint**：`curric_k1to6_12L_80000/step-60000.ckpt`（在 k=1..6 随机循环训练 60K step）
- **4D 隐状态**：`h` 形状是 `(B, S, F_groups, E)`，需要 `.reshape(B, -1, E).mean(1)` 才能输入 halt head

---

## 5. 基准数字

| 方案 | AUC（TabArena eval_bank） | mean_k |
|------|--------------------------|--------|
| loopk4 @ k=4（基准） | ≈ 0.810 | 4.0 |
| H1 置信度阈值 | 0.7981 | < 4 |
| H2 KL 稳定性 | 0.7974 | < 4 |
| H3 Halt MLP（step-10K） | 0.7867 | 1.00（退化） |
| Oracle（per-dataset 最优 k） | ≈ 0.893 | — |
| **H5/H6 目标** | **> 0.810** | **< 4** |

---

## 6. 后续实验步骤

### 6.1 训练完成后（80K steps）

```bash
# H5 eval（两个 lambda 变体）
CUDA_VISIBLE_DEVICES=0 python run_h5_eval.py \
    --ckpt $REPRO/ckpt/h5_act_l001_12L_80000/step-80000.ckpt

CUDA_VISIBLE_DEVICES=0 python run_h5_eval.py \
    --ckpt $REPRO/ckpt/h5_act_l005_12L_80000/step-80000.ckpt

# H6 eval
CUDA_VISIBLE_DEVICES=0 python run_h6_eval.py \
    --ckpt $REPRO/ckpt/h6_budget_cond_12L_80000/step-80000.ckpt
```

eval 脚本会输出 JSON 结果文件（默认 `$REPRO/h5_eval_results.json` 等），包含：
- 每个数据集的 AUC
- macro-averaged AUC
- mean_k（H5：来自 ponder cost；H6：来自 adaptive k 选择）

### 6.2 结果解读

**情形A：AUC > 0.810 且 mean_k < 4** → 成功，自适应 halting 有效  
**情形B：AUC > 0.810 但 mean_k ≈ 4** → 模型学会"不停"，ponder 惩罚太弱（H5：尝试更大 lambda；H6：需要不同的 k 选择策略）  
**情形C：AUC < 0.810** → fine-tuning 损伤了 backbone，考虑更低学习率或更短训练  
**情形D：AUC ≈ baseline 且 mean_k 任意** → H6 budget conditioning 没有带来增益，方向关闭

### 6.3 下一步候选方向（如果H5/H6也失败）

基于之前的分析，下一个架构方向是 **COCONUT / Ouro-style 隐状态推理**（不展开 token 序列，而是在 latent space 做多步 reasoning），或者 **TACO recipe pretraining**（直接在更大、更多样化的先验上预训练）。这两个方向预计比 adaptive halting 更根本地提升模型能力。

---

## 7. 文件索引

| 文件 | 用途 |
|------|------|
| `reproduce/exp/looped_pretrain/train_h5_act.py` | H5 ACT 训练脚本 |
| `reproduce/exp/looped_pretrain/train_h6_budget_cond.py` | H6 budget-cond 训练脚本 |
| `reproduce/exp/looped_pretrain/run_h5_eval.py` | H5 TabArena eval |
| `reproduce/exp/looped_pretrain/run_h6_eval.py` | H6 TabArena eval |
| `reproduce/exp/looped_pretrain/slurm_h5_l001.sh` | Slurm：H5 λ=0.01，2×H800 |
| `reproduce/exp/looped_pretrain/slurm_h5_l005.sh` | Slurm：H5 λ=0.05，2×H800 |
| `reproduce/exp/looped_pretrain/slurm_h6.sh` | Slurm：H6，4×H800 |
| `reproduce/exp/looped_pretrain/submit_all.sh` | 一键提交所有任务 |
| `reproduce/CLUSTER_MIGRATION.md` | 集群迁移操作指南（步骤细节） |
| `reproduce/eval_bank/` | TabArena eval 数据集（20 CSV，2.9MB） |
| `reproduce/ckpt/curric_k1to6_12L_80000/step-60000.ckpt` | Base checkpoint（84MB，GitHub Release 也有） |
