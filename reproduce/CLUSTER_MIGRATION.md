# 集群迁移指南：H5/H6 Slurm 8×H800

> 目标：把 H5-ACT (λ=0.01/0.05) 和 H6 Budget-Cond 三个训练任务迁移到 Slurm 8×H800 集群。  
> 预期耗时：~24–36h（对比本机单 L20 ~7 天）  
> **两台机器不互通，全部通过 GitHub 传输。**

---

## 0. 前提（已完成）

| 条件 | 状态 |
|------|------|
| 代码 + eval_bank 已 push | ✅ commit `1a93f44` — `git clone` 自带 eval_bank |
| DDP bug 已修 | ✅ commit `94e770f` |
| Slurm 脚本已 push | ✅ commit `7e7508b` |
| base checkpoint (84MB) | ✅ GitHub Release `v1.0-base-ckpt` |
| H6 本机进度 | step ~67K/80K，约2天完成，无需迁移 |

---

## 1. 在集群上执行（全部在集群登录节点）

### 1-a. clone 代码（含 eval_bank）

```bash
git clone https://github.com/zhongweixie/TabPFN_internal.git
cd TabPFN_internal
```

### 1-b. 下载 checkpoints（两个，都需要）

Release `v1.0-base-ckpt` 包含两个文件：
- `step-60000.ckpt` — base checkpoint，H5/H6 训练起点
- `step-80000.ckpt` — loopk4 reference，eval 时的外部基准（缺失则 ref 列全 NaN）

**方式一：gh CLI（推荐，一次下载全部）**
```bash
mkdir -p reproduce/ckpt/curric_k1to6_12L_80000
mkdir -p reproduce/ckpt/loopk4_12L_80000
# 下载 base checkpoint
gh release download v1.0-base-ckpt \
  --repo zhongweixie/TabPFN_internal \
  --dir reproduce/ckpt/curric_k1to6_12L_80000/ \
  --pattern "step-60000.ckpt"
# 下载 loopk4 reference checkpoint
gh release download v1.0-base-ckpt \
  --repo zhongweixie/TabPFN_internal \
  --dir reproduce/ckpt/loopk4_12L_80000/ \
  --pattern "step-80000.ckpt"
# 如果未登录：gh auth login
```

**方式二：wget + GitHub token（集群无 gh CLI 时）**
```bash
# 先在 GitHub Settings → Developer settings → Personal access tokens 生成 token（权限：repo read）
mkdir -p reproduce/ckpt/curric_k1to6_12L_80000 reproduce/ckpt/loopk4_12L_80000
wget --header="Authorization: token <YOUR_TOKEN>" \
  -O reproduce/ckpt/curric_k1to6_12L_80000/step-60000.ckpt \
  "https://github.com/zhongweixie/TabPFN_internal/releases/download/v1.0-base-ckpt/step-60000.ckpt"
wget --header="Authorization: token <YOUR_TOKEN>" \
  -O reproduce/ckpt/loopk4_12L_80000/step-80000.ckpt \
  "https://github.com/zhongweixie/TabPFN_internal/releases/download/v1.0-base-ckpt/step-80000.ckpt"
```

---

## 2. 验证文件完整性

```bash
# 检查文件大小（应约84MB）
ls -lh reproduce/ckpt/curric_k1to6_12L_80000/step-60000.ckpt

# 检查 eval_bank（应有20个CSV）
ls reproduce/eval_bank/ | wc -l

# 验证 checkpoint 可加载（需要 TACO 在 PYTHONPATH）
python -c "
import torch
ckpt = torch.load('reproduce/ckpt/curric_k1to6_12L_80000/step-60000.ckpt', map_location='cpu')
print('keys:', list(ckpt.keys())[:5])
print('OK')
"
```

---

## 3. 设置环境变量

```bash
# 加入 ~/.bashrc 或在提交前 export
export TACO_SRC="/path/to/TACO/src"       # TACO 代码库的 src 目录
export CONDA_ENV="wcb_rl_training"         # conda 环境名
export REPRO="$(pwd)/reproduce"            # 自动设为当前目录下的 reproduce
```

验证 TACO 可 import：
```bash
python -c "import sys; sys.path.insert(0, '$TACO_SRC'); import taco; print('TACO OK')"
```

---

## 4. 提交所有任务

```bash
cd reproduce/exp/looped_pretrain

# ⚠️ 必须先确认集群的 partition 名（各集群不同）
# 在登录节点运行：sinfo -s
# 然后编辑三个 slurm 脚本，在 #SBATCH 行里加上：
#   #SBATCH --partition=<your_partition>
#   #SBATCH --account=<your_account>   # 如需要

# dry-run 先验证路径无误
bash submit_all.sh --dry-run

# 正式提交（GPU分配：H5-l001×2 + H5-l005×2 + H6×4 = 8GPU）
# 注意：本机 H6 已跑到67K，集群只跑 H5 即可；或全提交让集群跑更快的 H6 覆盖
bash submit_all.sh
```

预期输出：
```
Submitted: slurm_h5_l001.sh
Submitted: slurm_h5_l005.sh
Submitted: slurm_h6.sh
Monitor jobs: squeue -u $USER
```

---

## 5. 监控进度

```bash
squeue -u $USER
tail -f h5_act_l001_<jobid>.out
tail -f h5_act_l005_<jobid>.out
```

| step | 预期 acc | 异常信号 |
|------|---------|---------|
| 1K | ~0.62 | < 0.55 重检配置 |
| 10K | ~0.65+ | loss 上升 → DDP 问题 |
| 20K | ~0.67+ | — |
| 80K | 目标 | — |

Checkpoint 保存：每 10K 一个永久 ckpt，每 5K 一个临时 ckpt。

---

## 6. 训练完成后跑 eval

```bash
# 如果是新的 shell session，先重新 export（训练时设置的变量不会自动保留）
export TACO_SRC="/path/to/TACO/src"
export REPRO="/path/to/TabPFN_internal/reproduce"

cd reproduce/exp/looped_pretrain

CUDA_VISIBLE_DEVICES=0 python run_h5_eval.py \
    --ckpt $REPRO/ckpt/h5_act_l001_12L_80000/step-80000.ckpt

CUDA_VISIBLE_DEVICES=0 python run_h5_eval.py \
    --ckpt $REPRO/ckpt/h5_act_l005_12L_80000/step-80000.ckpt

CUDA_VISIBLE_DEVICES=0 python run_h6_eval.py \
    --ckpt $REPRO/ckpt/h6_budget_cond_12L_80000/step-80000.ckpt
```

目标：AUC > 0.810（loopk4 基准），mean_k < 4，oracle 上界 0.893。

---

## 7. 本机 H6 状态

H6 在本机 GPU6 跑到 step ~67K（pid=3021044），约2天完成，不需要迁移。  
集群任务成功后可选择 kill：`kill 3021044`

---

## 注意事项

1. **WandB**：脚本默认 `--wandb_log True`，集群需配置 `WANDB_API_KEY`，否则改为 `--wandb_log False`。
2. **H6 eval** 每次 predict 必须包在 `enable_budget(k)/disable_budget()` 里（`run_h6_eval.py` 已修复）。
3. **GitHub Release URL**：`https://github.com/zhongweixie/TabPFN_internal/releases/tag/v1.0-base-ckpt`
