# TabPFN-3 论文实验复现

本目录包含 TabPFN-3 论文 (`paper_table_pfn_v3.md`) 核心宣称的复现实验：脚本、运行日志、验证报告。

**硬件**: 单卡 NVIDIA L20 (48GB) · **模型**: tabpfn 8.0.8 (本地 checkpoint) · **环境**: conda base

---

## 📊 看结果从这里开始

| 报告 | 内容 | 实验批次 |
|------|------|----------|
| [UPDATED_FINAL_SUMMARY.txt](reports/UPDATED_FINAL_SUMMARY.txt) | **总汇总**：全部已验证实验清单、覆盖率统计 | 汇总 (24实验时点) |
| [TOP3_HIGH_VALUE_REPORT.md](reports/TOP3_HIGH_VALUE_REPORT.md) | 时序预测 / SHAP+KV cache / 大数据扩展到1M | 第三批 (最新) |
| [HIGH_VALUE_EXPERIMENTS_REPORT.md](reports/HIGH_VALUE_EXPERIMENTS_REPORT.md) | Many-Class / 完整TabArena / 分位数回归 | 第二批 |
| [HIGH_PRIORITY_EXPERIMENTS_REPORT.md](reports/HIGH_PRIORITY_EXPERIMENTS_REPORT.md) | 因果推理 / TabArena子集 / V2.5 vs V3 | 第一批 |
| [PAPER_VS_VERIFICATION_ANALYSIS.md](reports/PAPER_VS_VERIFICATION_ANALYSIS.md) | 论文宣称 vs 实测结果逐条对比分析 | 对比分析 |
| [REMAINING_VERIFICATION_PLAN.md](reports/REMAINING_VERIFICATION_PLAN.md) | 尚未覆盖的实验与验证计划 | 待办清单 |

---

## 🔬 实验脚本 (scripts)

| 脚本 | 论文章节 | 日志 |
|------|----------|------|
| `run_large_data_scaling.py` | §3.2.1 大数据 (到1M行) | `large_data_scaling_results.log` |
| `run_shap_kv_cache.py` | §2.4.5 SHAP + KV cache | `shap_kv_cache_results.log` |
| `run_timeseries_fev.py` | §3.3 时序预测 (fev-bench) | `timeseries_fev_results.log` |
| `run_many_class_benchmark.py` | Many-Class decoder | `many_class_benchmark_results.log` |
| `run_full_tabarena.py` | 完整 TabArena (51数据集) | `full_tabarena_results.log` |
| `run_tabarena_subset.py` | TabArena 子集 | `tabarena_subset_results.log` |
| `run_quantile_regression_extended.py` | §3.2.4 分位数回归 | `quantile_regression_extended_results.log` |
| `run_causal_inference_v2.py` | §3.5 因果推理 | `causal_inference_v2_results.log` |
| `run_v2_5_vs_v3_comparison.py` | V2.5 vs V3 对比 | `v2_5_vs_v3_results.log` |
| `run_deep_verification.py` | 扩展性/高维深度验证 | `deep_verification_results.log` |
| `run_high_priority_experiments.py` | 第一批批量运行 | `high_priority_results.log` |
| `run_remaining_experiments.py` | 补充批量运行 | `remaining_experiments_results.log` |
| `run_supplementary_experiments.py` | 补充实验 | `supplementary_results.log` |
| `run_all_experiments.py` | 早期批量入口 | `experiment_full_results.log` |
| `01_basic_classification.py` | 基础分类 (依赖 `utils.py`) | — |
| `04_kv_cache_speedup.py` | KV cache 加速 (依赖 `utils.py`) | — |
| `experiment_07_high_dimensional.py` | 高维实验 | — |
| `test_simple.py` / `utils.py` | 冒烟测试 / 公共工具 | — |

---

## 📁 目录结构

```
reproduce/
├── README.md                  ← 本文件 (总索引入口)
├── run_*.py, utils.py         ← 实验脚本
├── *.log                      ← 运行日志
├── fev_bench_tasks.yaml       ← fev-bench 官方100任务定义 (run_timeseries_fev.py 读取)
├── embeddings_visualization.png
└── reports/                   ← 所有验证报告
    ├── *_REPORT.md / *_SUMMARY ← 有效报告 (见上方表格)
    └── archive/               ← 过时的迭代版本报告 (已被有效报告取代, 保留备查)
```

> **reports/archive/ 说明**: 整理时把多轮迭代产生的旧版总结/清单/索引移入此处 (进度数字 11→15→21 的早期快照, 已被 `reports/UPDATED_FINAL_SUMMARY.txt` 等当前版本覆盖)。内容未删除, 仅归档。

---

## ⚙️ 复现要点

- 运行用 conda base: `/home/zxiebk/miniconda3/bin/python`
- 模型 checkpoint: `/home/zxiebk/workspace/model/tabpfn_3/`
- License 绕过: 脚本头部 `os.environ["TABPFN_NO_BROWSER"] = "0"`
- 时序额外依赖: `tabpfn-time-series`, `shapiq`, `tabpfn-extensions`, `autogluon.timeseries`, `fev`
- 时序 checkpoint 需软链到 `~/.cache/tabpfn/` 供 LOCAL 模式加载
