# TabPFN-3 论文未验证实验详细清单

基于完整论文分析，以下是所有未验证的实验。

---

## 📊 实验章节总览

### Section 2: 架构和优化
- ✅ 2.1 Architecture - 基本功能已验证
- ✅ 2.2 Many-Class Decoder - 已验证50类
- ⚠️ 2.3 Preprocessing - 部分验证
- ⚠️ 2.4 Inference Optimization - 部分验证
- ❌ 2.5 Synthetic Prior - 无法验证（训练相关）
- ❌ 2.6 TabPFN-3-Plus - 仅API可用

### Section 3: 实验结果
- ❌ 3.1.1 TabArena - 需要51个数据集
- ❌ 3.1.2 TALENT - 需要274个数据集
- ❌ 3.1.3 TabSTAR - 需要文本特征支持
- ⚠️ 3.2.1 Large Data - 部分验证（50K，论文1M）
- ✅ 3.2.2 Many-Class - 已验证
- ⚠️ 3.2.3 Many Features - 受限于2000特征上限
- ⚠️ 3.2.4 Quantile Regression - 基础验证
- ❌ 3.3 Time-Series - 需要TabPFN-TS-3
- ❌ 3.4 Relational Data - 需要多表数据
- ❌ 3.5 Causal Inference - 需要专门数据集
- ⚠️ 3.6 Embeddings - 可测试但未执行

---

## 🔴 完全未验证的实验

### 1. TabArena基准测试 (Section 3.1.1)
**论文内容**:
- 51个精选数据集
- TabPFN-3领先72 Elo points
- 10k-100k行子集领先100 Elo
- Win rate: 80% vs tuned GBTs
- Figure 10, 11, 12

**为什么未验证**: 需要下载和准备51个OpenML数据集

**可行性**: ⚠️ 中等 - 数据集公开但需要大量准备工作

### 2. TALENT基准测试 (Section 3.1.2)
**论文内容**:
- 274个多样化数据集
- TabPFN-3排名第一
- Figure 13, 28

**为什么未验证**: 需要300个数据集（减去26个开发集）

**可行性**: ⚠️ 中等 - 数据集公开但工作量巨大

### 3. TabSTAR文本-表格基准 (Section 3.1.3)
**论文内容**:
- 50个文本-表格数据集
- TabPFN-3-Plus显著领先
- Figure 14

**为什么未验证**: 
- 需要TabPFN-3-Plus（仅API）
- 需要文本特征支持

**可行性**: ❌ 低 - 需要API访问

### 4. 大规模数据基准 (Section 3.2.1, 100K-1M行)
**论文内容**:
- 13个数据集（9分类 + 4回归）
- 100K-1M行，最多200特征
- 超越8小时调优的GBT基线
- Figure 15, 16, 30

**为什么未验证**: 
- GPU内存限制（仅验证到50K）
- 需要专有大型数据集

**可行性**: ⚠️ 中等 - 可以生成合成数据但需要更多GPU内存

**已完成**: ✅ 50K样本验证

### 5. 时间序列预测 (Section 3.3)
**论文内容**:
- TabPFN-TS-3在fev-bench上排名第2
- 100个时间序列任务
- SQL skill: 43.1%, MASE skill: 30.6%
- 支持32K历史时间步
- Table 1, Figure 20

**为什么未验证**: 需要TabPFN-TS-3专门模型

**可行性**: ❌ 低 - 需要专门的时间序列模型和库

### 6. 关系数据 (Section 3.4)
**论文内容**:
- RelBenchV1基准测试
- TabPFN-REL在RFM中SOTA
- Figure 21

**为什么未验证**: 
- 需要多表关系数据
- 需要RDBLearn集成

**可行性**: ❌ 低 - 需要复杂的多表数据集

### 7. 因果推理 (Section 3.5)
**论文内容**:
- RealCause和scikit-uplift基准
- T/X/S meta-learners
- QINI-score评估
- Figure 27

**为什么未验证**: 需要因果推理专门数据集

**可行性**: ⚠️ 中等 - 数据集可能公开

### 8. TabPFN-3-Plus (Thinking Mode) (Section 2.6)
**论文内容**:
- Native text-feature support
- Test-time compute scaling
- 领先非TabPFN模型200+ Elo
- 领先AutoGluon 1.5 extreme 100+ Elo

**为什么未验证**: 仅通过API和企业许可可用

**可行性**: ❌ 不可能 - 闭源

---

## ⚠️ 部分验证的实验

### 9. Row-Chunking内存优化 (Section 2.4.1)
**论文内容**:
- 两阶段推理方案
- 峰值内存与数据集大小解耦
- 5x内存减少
- Figure 6

**已验证**: 功能存在且工作
**未验证**: 
- 详细内存profiling
- 不同数据规模的内存对比
- 时间开销量化

**可行性**: ✅ 高 - 可以添加详细内存监控

### 10. torch.compile加速 (Section 2.4.4)
**论文内容**:
- 最高1.58x加速
- 与row chunking兼容
- Appendix G.1

**已验证**: 无
**可行性**: ✅ 高 - 容易测试

**建议实验**:
```python
clf = TabPFNClassifier(device="cuda", model_path=MODEL_PATH)
# 测试compile前后速度
```

### 11. FlashAttention-3 (Section 2.4.4)
**论文内容**:
- 1.5-1.7x加速 @ 1M rows
- 需要H100 + Hopper架构

**已验证**: 无
**可行性**: ❌ 低 - 需要H100 GPU（当前是L20）

### 12. 高维特征 (Section 3.2.3)
**论文内容**:
- 100-320样本，1,100-22,200特征
- TabPFN-3 with 32 estimators最佳
- Figure 18

**已验证**: 测试到2000特征（模型上限）
**未验证**: 无法测试>2000特征

**可行性**: ❌ 低 - 受模型限制

### 13. 模型蒸馏 (Section 2.4.3)
**论文内容**:
- 蒸馏到MLP或tree ensembles
- CPU上sub-millisecond延迟
- 保留大部分性能

**已验证**: 无
**可行性**: ⚠️ 中等 - 需要企业版功能

### 14. Embeddings提取 (Section 3.6)
**论文内容**:
- 提取语义有意义的行嵌入
- ICL层输出
- Figure 22展示PCA可视化

**已验证**: 无
**可行性**: ✅ 高 - TabPFN有get_embeddings方法

**建议实验**:
```python
embeddings = clf.get_embeddings(X_test, data_source="test")
# PCA可视化
```

---

## 🔢 未验证的Figure/Table清单

### Figures (论文中约40+个图)

**已验证相关**:
- ✅ Figure 7-8: KV Cache加速 - 已验证
- ✅ Figure 11: Pareto前沿 (estimator数量) - 已验证

**完全未验证**:
- ❌ Figure 1-3: TabArena结果
- ❌ Figure 4b: 版本对比
- ❌ Figure 6: Row-chunking内存
- ❌ Figure 9: SCM prior可视化
- ❌ Figure 10: TabArena性能
- ❌ Figure 12: Win rate矩阵
- ❌ Figure 13: TALENT排名
- ❌ Figure 14: TabSTAR结果
- ❌ Figure 15-16: Large data基准
- ❌ Figure 17: Many-class合成基准
- ❌ Figure 18: Many features
- ❌ Figure 19: Quantile regression
- ❌ Figure 20: 时间序列预测
- ❌ Figure 21: RelBench关系数据
- ❌ Figure 22: Embeddings可视化
- ❌ Figure 27: 因果推理

### Tables

**已验证相关**:
- ✅ Table 相关内容: 基础性能指标

**未验证**:
- ❌ Table 1: fev-bench时间序列
- ❌ Appendix中的详细表格

---

## 📊 验证统计

### 已验证
- **核心功能**: 6/6 (100%)
- **补充功能**: 5/5 (100%)
- **Section 2架构**: 50%
- **Section 3实验**: 20%
- **总体论文覆盖**: ~25%

### 未验证原因分类
1. **需要专门模型/API** (35%): TabPFN-3-Plus, TabPFN-TS-3
2. **需要大量数据集** (30%): TabArena, TALENT, RelBench
3. **硬件限制** (15%): FlashAttention-3, >50K样本
4. **模型限制** (10%): >2000特征
5. **时间/资源限制** (10%): 完整基准测试

---

## 🎯 可以补充验证的实验（优先级）

### 🔥 高优先级（容易且重要）

1. **Embeddings提取和可视化**
   - 时间: 10分钟
   - 难度: 低
   - 价值: 高（验证语义表示）

2. **torch.compile加速测试**
   - 时间: 15分钟
   - 难度: 低
   - 价值: 中（性能优化验证）

3. **详细内存profiling**
   - 时间: 20分钟
   - 难度: 中
   - 价值: 高（验证内存优化）

4. **模型版本对比** (V2.6 vs V3)
   - 时间: 30分钟
   - 难度: 低
   - 价值: 高（验证改进幅度）

### 📊 中优先级（需要更多资源）

5. **更大规模测试** (尝试100K)
   - 时间: 30分钟
   - 难度: 中
   - 价值: 高（推进扩展性验证）

6. **TabArena部分数据集**
   - 时间: 2小时
   - 难度: 中
   - 价值: 高（真实基准）

7. **因果推理数据集**
   - 时间: 1小时
   - 难度: 中
   - 价值: 中（扩展应用）

### 🔴 低优先级（限制太多）

8. FlashAttention-3 - 需要H100
9. 完整TALENT - 太多数据集
10. 关系数据 - 需要复杂数据

---

## 💡 建议的下一步行动

如果要继续实验，建议按此顺序：

1. **Embeddings可视化** - 快速且有价值
2. **torch.compile加速** - 验证优化选项
3. **模型版本对比** - 验证改进声明
4. **详细内存profiling** - 完善row-chunking验证
5. **尝试100K样本** - 推进扩展性上限

这些可以在1-2小时内完成，显著提升论文覆盖率。

---

## 📈 如果完成高优先级实验

**预期论文覆盖率**: 25% → 35%

**预期验证指标**:
- Section 2架构: 50% → 70%
- Section 3实验: 20% → 25%
- 关键技术声明: 9/9 → 12/15

---

**总结**: 已验证的实验覆盖了论文的核心技术声明（KV Cache、Many-class、扩展性、性能）。未验证的主要是需要大量数据集的基准测试、专门模型的扩展应用、以及硬件受限的优化特性。
