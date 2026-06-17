# TabPFN-3 未验证实验清单

根据论文分析，以下是还未验证的关键实验：

## 📋 未验证实验清单

### 1. ⚠️ **高维特征测试** (Section 3.2.3)
**论文内容**:
- 测试100-320样本，1,100-22,200特征的场景
- TabPFN-3每个estimator最多200特征
- 使用32个estimators达到最佳性能

**为什么重要**: 测试高维低样本场景的泛化能力

**可复现性**: ✅ 可以生成合成数据测试

### 2. ⚠️ **分位数回归** (Section 3.2.4)
**论文内容**:
- TabPFN-3通过bar-distribution回归头提供完整预测分布
- 单次前向传播可解码任意分位数
- 在10个分位数水平上测试 (0.1, 0.2, ..., 0.9)
- 归一化pinball loss接近1.00

**为什么重要**: 验证概率预测能力

**可复现性**: ✅ 可以使用TabPFNRegressor测试

### 3. ⚠️ **Row-Chunking内存优化** (Section 2.4.1)
**论文内容**:
- 两阶段推理方案
- 峰值内存与数据集大小解耦
- 5x内存减少，仅几个百分点的时间开销
- Figure 6展示了内存-时间权衡

**为什么重要**: 验证内存优化关键技术

**可复现性**: ✅ 需要测试大数据集的内存使用

### 4. ⚠️ **torch.compile和FlashAttention-3** (Section 2.4.4)
**论文内容**:
- torch.compile: 最高1.58x加速
- FlashAttention-3: 1.5-1.7x加速 @ 1M rows (仅H100)
- 与row chunking兼容

**为什么重要**: 验证性能优化选项

**可复现性**: ⚠️ FA3需要H100，但可以测试torch.compile

### 5. ⚠️ **不同estimator数量对比** (Figure 11)
**论文内容**:
- N1, N2, N4: 1, 2, 4个estimators
- Pareto前沿分析：质量 vs 时间权衡

**为什么重要**: 理解estimator数量的影响

**可复现性**: ✅ 易于测试

### 6. ⚠️ **Missing Values处理** (Section 2.1)
**论文内容**:
- Native missing-value handling
- NaN indicator连接到cell value
- 模型直接处理缺失数据

**为什么重要**: 验证缺失值处理能力

**可复现性**: ✅ 可以注入NaN测试

### 7. ⚠️ **Categorical Features处理**
**论文内容**:
- 支持类别特征
- 预处理pipeline自动处理

**为什么重要**: 验证混合数据类型支持

**可复现性**: ✅ 可以创建混合数据集

### 8. ⚠️ **Out-of-Distribution泛化** (Section 2.5)
**论文内容**:
- OOD prior训练
- 从插值到外推的能力
- Figure 26展示外推能力

**为什么重要**: 验证分布外泛化

**可复现性**: ✅ 可以创建分布偏移数据集

### 9. ⚠️ **大规模测试** (100K-1M行)
**论文内容**:
- Figure 15: 在100K-1M行数据上的性能
- Figure 16: 扩展曲线 (100K, 250K, 500K, 1M)
- 超越8小时调优的GBT基线

**为什么重要**: 验证最大扩展能力

**可复现性**: ⚠️ 需要大量GPU内存和时间

### 10. ⚠️ **模型版本对比** (V2 vs V2.5 vs V2.6 vs V3)
**论文内容**:
- Figure 4b: 各版本per-dataset分数对比
- TabPFN-3比TabPFN-2.5提升72 Elo points
- 速度提升最多20x

**为什么重要**: 验证版本改进

**可复现性**: ✅ 可以加载不同版本模型

### 11. ❌ **时间序列预测** (Section 3.3)
**论文内容**:
- TabPFN-TS-3在fev-bench上排名第2
- 需要tabpfn-time-series库

**为什么重要**: 验证时间序列能力

**可复现性**: ❌ 需要专门的时间序列模型

### 12. ❌ **关系数据** (Section 3.4)
**论文内容**:
- RelBenchV1基准测试
- 需要多表数据

**可复现性**: ❌ 需要专门数据集

### 13. ❌ **TabArena完整基准** (Section 3.1.1)
**可复现性**: ❌ 需要51个公开数据集

### 14. ❌ **TALENT基准** (Section 3.1.2)
**可复现性**: ❌ 需要274个数据集

## 🎯 优先级排序

### 🔥 高优先级（易复现且重要）
1. ✅ **高维特征测试** - 验证特征选择能力
2. ✅ **分位数回归** - 验证概率预测
3. ✅ **不同estimator数量** - 理解性能权衡
4. ✅ **Missing Values处理** - 实用功能
5. ✅ **Categorical Features** - 混合数据支持
6. ✅ **模型版本对比** - 验证改进

### 📊 中优先级（需要资源）
7. ⚠️ **Row-Chunking内存** - 需要大数据集
8. ⚠️ **torch.compile** - 性能优化
9. ⚠️ **OOD泛化** - 分布偏移测试

### 🔴 低优先级（资源限制）
10. ❌ **大规模测试 (1M)** - GPU内存限制
11. ❌ **时间序列** - 需要专门模型
12. ❌ **完整基准** - 需要大量数据集

## 📝 建议的补充实验

基于以上分析，建议执行以下补充实验：

1. **experiment_07_high_dimensional.py** - 高维特征测试
2. **experiment_08_quantile_regression.py** - 分位数回归
3. **experiment_09_estimators_comparison.py** - Estimator数量对比
4. **experiment_10_missing_values.py** - 缺失值处理
5. **experiment_11_categorical_features.py** - 类别特征测试
6. **experiment_12_ood_generalization.py** - OOD泛化
7. **experiment_13_model_versions.py** - 版本对比
