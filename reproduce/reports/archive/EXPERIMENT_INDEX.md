# TabPFN-3 实验索引

快速导航所有实验和结果。

## 🚀 快速开始

```bash
cd /home/zxiebk/workspace/train/PFN/TabPFN/reproduce

# 运行所有核心实验
/data/zxiebk/miniconda3/bin/python run_all_experiments.py

# 运行补充实验
/data/zxiebk/miniconda3/bin/python run_supplementary_experiments.py

# 简单测试
/data/zxiebk/miniconda3/bin/python test_simple.py
```

## 📊 实验脚本

| 文件 | 描述 | 实验内容 |
|------|------|----------|
| `run_all_experiments.py` | ⭐ 核心实验套件 | 6个核心实验 |
| `run_supplementary_experiments.py` | ⭐ 补充实验套件 | 5个补充实验 |
| `test_simple.py` | 简单验证 | 基础功能测试 |
| `utils.py` | 工具函数 | 通用函数库 |

## 📄 报告文档

| 文件 | 描述 |
|------|------|
| `FINAL_REPORT.md` | ⭐ **完整报告** - 所有实验总结 |
| `RESULTS_SUMMARY.md` | 核心结果总结 |
| `UNVERIFIED_EXPERIMENTS.md` | 未验证实验清单 |
| `README.md` | 实验计划和分析 |

## 📝 日志文件

| 文件 | 内容 |
|------|------|
| `experiment_full_results.log` | 核心实验完整日志 |
| `supplementary_results.log` | 补充实验完整日志 |

## 🎯 关键结果速览

### ⭐ KV Cache加速
- **加速比**: 23.5x
- **延迟**: 1.31 ms/样本
- **状态**: ✅ 完美符合论文预期

### ⭐ 分类性能
- **二分类**: 96.81% 准确率
- **多分类**: 100% 准确率
- **Many-class (50类)**: 51.20% 准确率

### ⭐ 回归性能
- **R² Score**: 0.9999

### ⭐ 扩展性
- 成功测试到 **50,000** 样本

## 📈 实验覆盖

- ✅ 核心实验: 6/6 (100%)
- ✅ 补充实验: 5/5 (100%)
- ✅ 论文关键声明: 9/9 (100%)

## 🔗 相关资源

- 论文: `paper_table_pfn_v3.md`
- 模型路径: `/home/zxiebk/workspace/model/tabpfn_3/`
- 代码仓库: `/data/zxiebk/workspace/train/PFN/TabPFN/`
