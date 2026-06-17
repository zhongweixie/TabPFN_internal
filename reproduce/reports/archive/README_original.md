# TabPFN-3 实验复现计划

本目录包含TabPFN-3论文实验的复现脚本。

## 论文关键实验分析

根据 `paper_table_pfn_v3.md` 分析，论文的核心实验包括：

### 1. 架构创新验证
- **三阶段架构**: Feature Distribution Embedding → Feature Aggregation → ICL
- **Many-Class Decoder**: 非参数化多类别解码器，支持任意数量类别
- **Row-Chunking**: 内存优化，支持1M行数据
- **KV Cache**: 1-3个数量级的推理加速

### 2. 公开基准测试 (Section 3.1)

#### 3.1.1 TabArena (51个数据集，最多100K行)
- **关键结果**:
  - TabPFN-3 在单次前向传播中超越所有模型
  - 比TabPFN-2.5提升 72 Elo points
  - TabPFN-3 vs tuned XGBoost/LightGBM/CatBoost: 80% win rate
  - 在10k-100k行子集上领先100 Elo

#### 3.1.2 TALENT (274个数据集)
- TabPFN-3 在aggregate和各任务类型上排名第一

#### 3.1.3 TabSTAR (50个文本-表格数据集)
- TabPFN-3-Plus (Thinking) 显著领先
- TabPFN-3 在数值模型中表现最佳

### 3. 内部基准测试 (Section 3.2)

#### 3.2.1 Large Data (100K-1M行，200特征)
- **13个数据集**: 9个分类 + 4个回归
- TabPFN-3 超越8小时调优的GBT基线
- 在1M行规模保持SOTA性能

#### 3.2.2 Many-Class Classification
- 支持任意数量类别（论文训练最多1024类）

#### 3.2.3 High-Dimensional Features
- 测试高特征维度场景

#### 3.2.4 Quantile Regression
- 分位数回归任务

### 4. 性能优化验证 (Section 2.4)

#### 2.4.1 Row-Chunking
- Figure 6: 峰值内存与行数/特征数的关系
- 5x内存减少，仅数个百分点的时间开销

#### 2.4.2 KV Cache
- Figure 7-8: 
  - 7GB KV cache per estimator @ 1M rows
  - 1-3个数量级的预测加速
  - 0.1-3 ms/测试点 @ batch=100

#### 2.4.3 Compilation + FlashAttention-3
- torch.compile: 最高1.58x加速
- FA3: 1.5-1.7x加速 @ 1M rows

### 5. 时间序列和关系数据 (Section 3.3, 3.4)
- TabPFN-TS-3 在时间序列基准上排名第2
- RelBench关系数据SOTA

## 可复现的实验

基于现有模型和数据，我们可以复现：

### ✅ 可以复现的实验

1. **基础分类和回归** (examples中已有)
2. **KV Cache加速测试** (examples/kv_cache_fast_prediction.py)
3. **扩展性测试** (1K → 100K样本)
4. **Many-Class分类** (50-100类)
5. **性能对比** (V2.6 vs V3)
6. **内存和时间profiling**
7. **不同fit_mode对比**

### ⚠️ 需要额外数据的实验

1. **TabArena基准** - 需要51个数据集
2. **TALENT基准** - 需要274个数据集
3. **Large Data基准** - 需要专有数据集（100K-1M行）

### ❌ 无法复现的实验

1. **TabPFN-3-Plus (Thinking)** - 仅API可用
2. **TabSTAR** - 需要文本特征支持

## 实验脚本结构

```
reproduce/
├── README.md                           # 本文件
├── 01_basic_classification.py          # 基础分类测试
├── 02_basic_regression.py              # 基础回归测试
├── 03_many_class_classification.py     # 多类别分类
├── 04_kv_cache_speedup.py              # KV缓存加速
├── 05_scalability_test.py              # 扩展性测试
├── 06_memory_profiling.py              # 内存分析
├── 07_version_comparison.py            # 版本对比
├── 08_fit_mode_comparison.py           # fit_mode对比
└── utils.py                            # 工具函数
```

## 运行环境

- Python 3.12+
- PyTorch 2.8+
- TabPFN 8.0.8
- NVIDIA L20 GPU (46GB)
- 模型路径: `/home/zxiebk/workspace/model/tabpfn_3/`

## 使用方法

```bash
# 运行所有实验
cd /home/zxiebk/workspace/train/PFN/TabPFN/reproduce
python run_all.py

# 运行单个实验
python 01_basic_classification.py
python 04_kv_cache_speedup.py
```

## 预期结果

根据论文，预期复现的关键指标：

1. **分类准确率**: 在标准数据集上达到SOTA水平
2. **KV Cache加速**: 1-3个数量级的预测加速
3. **扩展性**: 支持100K行数据，单GPU推理
4. **内存效率**: Row-chunking降低峰值内存
5. **速度**: 比TabPFN-2.5快最多20倍
