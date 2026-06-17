#!/usr/bin/env python3
"""
TabPFN-TS-3 时序预测验证 (Section 3.3) - fev-bench 子集

论文宣称:
  - 纯合成数据训练, 零真实时序数据
  - fev-bench: SQL skill 43.1%, MASE skill 30.6%, 总排名 #2

本实验:
  1. 用本地 timeseries checkpoint (tabpfn-v3-regressor-v3_20260506_timeseries.ckpt)
  2. 在 fev-bench 官方 100 任务的子集上运行 (默认前 N 个)
  3. 计算每任务 SQL/MASE, 并相对 Seasonal Naive 基线计算 skill score
     skill = 1 - model_error / seasonal_naive_error  (越高越好, >0 优于基线)
  4. 与论文 skill 数字对照 (诚实说明: 论文 skill 相对完整 30+ 模型 leaderboard 聚合,
     此处相对 Seasonal Naive, 用于验证模型确有强时序能力的趋势)

用法: python run_timeseries_fev.py [N_TASKS]
"""

import os
os.environ["TABPFN_NO_BROWSER"] = "0"

import sys
import time
import numpy as np

import fev
from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode

YAML = "fev_bench_tasks.yaml"
N_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 12

print("=" * 80)
print("TabPFN-TS-3 时序预测验证 (Section 3.3) - fev-bench")
print("=" * 80)
print(f"\n论文宣称: SQL skill 43.1%, MASE skill 30.6%, 排名 #2 (纯合成数据训练)")
print(f"运行任务数: {N_TASKS} (共100)")

bench = fev.Benchmark.from_yaml(YAML)
# 真实计算成本 = 时序条数 (series) * num_windows。
# fev-bench 任务的主导成本是 series 数量 (8 ~ 30490), 而非 horizon。
# 用一组已测量的轻量任务白名单 (series*windows 最小), 保证在预算内完成。
LIGHT_TASKS = [
    "uk_covid_nation_1W/new",          # 4 series x 4 win
    "uk_covid_nation_1W/cumulative",   # 4 series x 4 win
    "rohlik_orders_1W",                # 7 series x 5 win
    "australian_tourism",              # 89 series x 2 win
    "world_tourism",                   # 178 series x 2 win
    "world_co2_emissions",             # 191 series x 9 win
]
_by_name = {t.task_name: t for t in bench.tasks}
tasks = [_by_name[n] for n in LIGHT_TASKS if n in _by_name][:N_TASKS]
print(f"已加载 {len(bench.tasks)} 个任务, 运行轻量白名单 {len(tasks)} 个")
print("选中任务: " + ", ".join(f"{t.task_name}(h{t.horizon}xw{t.num_windows})" for t in tasks))

# 构建一次 pipeline (复用)
print("\n构建 TabPFN-TS-3 pipeline (LOCAL)...")
pipe = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL, tabpfn_output_selection="median")
print("pipeline 就绪")


def seasonal_naive_baseline(task):
    """对该任务计算 Seasonal Naive 基线的 SQL/MASE, 作为 skill 参照。
    用 fev 内置的简单基线: 重复最后一个季节周期。返回 evaluation summary。"""
    import datasets
    season = task.seasonality or 1
    quantiles = task.quantile_levels or [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    preds_per_window = []
    for window in task.iter_windows():
        past, future = fev.convert_input_data(window, adapter="autogluon", as_univariate=True)
        h = task.horizon
        # 每个 item 取最后 season 个值循环填充 horizon
        out_rows = []
        for item_id, g in past.groupby(level=0):
            vals = g["target"].values
            if len(vals) >= season and season > 0:
                tail = vals[-season:]
            else:
                tail = vals[-1:] if len(vals) else np.array([0.0])
            fc = np.resize(tail, h)
            row = {"predictions": fc.tolist()}
            for q in quantiles:
                row[str(q)] = fc.tolist()  # 点基线: 分位数退化为点预测
            out_rows.append(row)
        ds = datasets.Dataset.from_list(out_rows)
        preds_per_window.append(
            fev.utils.combine_univariate_predictions_to_multivariate(
                ds, target_columns=task.target_columns
            )
        )
    return task.evaluation_summary(preds_per_window, model_name="SeasonalNaive")


results = []
for i, task in enumerate(tasks):
    name = task.task_name
    print(f"\n[{i+1}/{len(tasks)}] {name} | metric={task.eval_metric} | h={task.horizon} | windows={task.num_windows}")
    rec = {"task": name, "metric": task.eval_metric}
    try:
        t0 = time.time()
        preds, inf_s = pipe.predict_fev(task, use_covariates=True)
        summary = task.evaluation_summary(preds, model_name="TabPFN-TS-3", inference_time_s=inf_s)
        wall = time.time() - t0

        model_err = summary.get("test_error")
        # 各 metric 的绝对值也可能在 summary 中
        rec["test_error"] = model_err
        rec["inference_s"] = round(inf_s, 1)
        rec["wall_s"] = round(wall, 1)

        # 基线
        base = seasonal_naive_baseline(task)
        base_err = base.get("test_error")
        rec["baseline_error"] = base_err

        if model_err is not None and base_err not in (None, 0):
            skill = 1.0 - (model_err / base_err)
            rec["skill"] = skill
            print(f"    {task.eval_metric}: model={model_err:.4f} | SeasonalNaive={base_err:.4f} | skill={skill:+.1%}")
        else:
            print(f"    {task.eval_metric}: model={model_err} | baseline={base_err}")
        print(f"    推理 {inf_s:.1f}s | 总 {wall:.1f}s")
        rec["success"] = True
    except Exception as e:
        import traceback
        rec["success"] = False
        rec["error"] = str(e)[:200]
        print(f"    ✗ 失败: {str(e)[:160]}")
        traceback.print_exc()
    results.append(rec)

# ============================================================================
# 汇总
# ============================================================================
print("\n" + "=" * 80)
print("汇总: fev-bench 子集结果")
print("=" * 80)

ok = [r for r in results if r.get("success")]
skills = [r["skill"] for r in ok if "skill" in r]
beat = [s for s in skills if s > 0]

print(f"\n{'任务':<28} | {'metric':>6} | {'model_err':>10} | {'baseline':>10} | {'skill':>8}")
print("-" * 78)
for r in results:
    if r.get("success") and "skill" in r:
        print(f"{r['task'][:28]:<28} | {r['metric']:>6} | {r['test_error']:>10.4f} | "
              f"{r['baseline_error']:>10.4f} | {r['skill']:>+8.1%}")
    elif r.get("success"):
        print(f"{r['task'][:28]:<28} | {r['metric']:>6} | {str(r.get('test_error')):>10} | {'—':>10} | {'—':>8}")
    else:
        print(f"{r['task'][:28]:<28} | {'FAIL':>6} | {'—':>10} | {'—':>10} | {'—':>8}")

if skills:
    print(f"\n完成任务: {len(ok)}/{len(tasks)}")
    print(f"相对 Seasonal Naive 的平均 skill: {np.mean(skills):+.1%}")
    print(f"优于基线的任务: {len(beat)}/{len(skills)} ({len(beat)/len(skills):.0%})")
    print(f"\n论文 (相对完整leaderboard聚合): SQL skill 43.1%, MASE skill 30.6%, 排名 #2")
    print(f"说明: 本 skill 相对 Seasonal Naive 单一基线, 数值不直接可比;")
    print(f"      正 skill 且高胜率即验证 '纯合成数据训练的模型具备强时序能力' 的核心宣称。")

print("\n" + "=" * 80)
print("验证完成")
print("=" * 80)
