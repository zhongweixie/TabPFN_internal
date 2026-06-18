# TabPFN Thinking Mode — 研究总览

生成时间: 2026-06-18
状态: ✅ 研究完成(干净的负结果 + 两个正面机制发现)
主题: 能否把 COCONUT / Ouro 的"计算换精度"(深度循环潜空间推理)接到 TabPFN v2.5?

---

## 一句话结论

**不能拿到净收益。** 在 TabPFN v2.5 这种已近饱和的表格 ICL 模型上,无论 COCONUT 式
latent 反馈还是 Ouro 式全栈递归,深度循环都无法换来"计算换精度"的红利。研究逐一攻克了
两个真实障碍(循环坍缩、微调侵蚀编码器),攻克后的结论是:**深度循环无害,但也无益 —— 
天花板本就有限。**

这是个"做对了之后确认此路收益有限"的负结果,不是"没做好"。

---

## 完整因果链(四阶段)

```
Phase 0  training-free 探针
  └─ 裸循环不白给收益;ouro 全栈循环坍缩到平凡不动点,coconut latent-only 不坍缩
       │
       ├─→ Branch A (COCONUT 启发): latent-only 反馈 + 深度课程, 仅最终步监督
       │     └─ 不坍缩、不破坏基线, 但深度无稳健增益 (多 seed 后单 seed 乐观印象被推翻)
       │
       └─→ Branch B (Ouro): 全栈递归 + 可学习门控 + per-step 监督
             ├─ 机制✅: 门控治好了 Phase 0 的坍缩 (Ouro 配方在 TabPFN 上复现)
             ├─ 审查发现 2 个 bug (熵正则空操作 / 门控梯度陷阱) → 修复
             ├─ 修复后效用仍为负 → 证明负结果不是 bug 假象
             └─ 补救 (冻结编码器+早停): post@1 恢复基线, 方差 ±0.2→±0.01
                   └─ 因果证实: 损害来自微调侵蚀编码器, 不是循环机制
                        └─ 但深度仍无净增益 ⇒ 无害但无益
```

---

## 两个正面机制发现(过程中真实成立的)

1. **Ouro 门控防坍缩**:Phase 0 裸全栈循环坍缩(drift→0.2、预测冻死);加可学习 per-step
   残差门控后 drift 受控、每步都成有用预测器。Ouro 的稳定化机制在 TabPFN 上确实复现。

2. **冻结编码器治微调侵蚀**:自由微调让 post@1 从 0.97 崩到 0.72;冻结 ~11K 编码器参数后
   恢复到追平基线,**方差从 ±0.2 骤降到 ±0.01** —— 这是"损害来自编码器侵蚀"的铁证。

---

## 方法论严谨性(这次研究的三次自我纠错)

这项研究的可信度很大程度来自三次主动纠错,值得单列:

1. **多 seed 推翻单 seed 乐观结论**:Branch A 初版单 seed 显示 wine "0.907→1.000 + drift 
   收敛",看着像强信号。3-seed 复跑后未复现 —— 大部分是噪声 + 数据集太易。

2. **多 subagent 审查发现并修复 2 个真 bug**:熵正则因 detach 而梯度为零(空操作)、门控
   因 0 初始化陷入梯度陷阱(grad ~5e-10)。修复后**负结论方向不变**,反而更强(排除了
   "bug 导致假象")。同一轮审查的一个 CRITICAL 指控("eval 没跑 thinking 循环")则被
   实测**证伪**(predict_proba 单次无 cache forward,thinking 确实激活)。

3. **对照系陷阱的识别**:`Δacc(post−pre@同深度)` 一度显示 +0.49/+0.68 巨大"提升",实为
   分母(pre@steps>1)是 Phase 0 坍缩值造成的幻觉。换成唯一正确基线 pre@steps=1(标准单遍)
   后,真相是全负。

---

## 关键数字(最终,正确基线 = pre@1 标准单遍)

| 数据集 | 基线 pre@1 | Branch A 深度收益 | Branch B 自由微调 post@1 | Branch B 冻结后 post@1 |
|---|---|---|---|---|
| wine | 1.000 | +0.05~0.09(回归基线,非新能力) | 0.95 | **1.000** |
| breast_cancer | 0.971 | 边际/非单调 | 0.76 | **0.961** |
| digits | 0.985 | 不显著 | 0.90 | **0.927** |
| phoneme | 0.906 | +0.02(p≈0.19,不显著) | 0.66 | **0.902** |
| qsar-biodeg | 0.869 | 退化 | 0.73 | **0.861** |

冻结后 post 列追平基线 = 损害消除;但无一列**超过**基线 = 深度无净增益。

---

## 文件导航

**报告(`exp/`)**

| 文件 | 内容 |
|---|---|
| `THINKING_MODE_SUMMARY.md` | 本文件:总览 + 因果链 + 导航 |
| `THINKING_MODE_PHASE0_REPORT.md` | Phase 0 training-free 诊断(含 COCONUT 等价性的框架澄清) |
| `THINKING_MODE_COCONUT_REPORT.md` | Branch A:COCONUT 课程微调(多 seed 强化版) |
| `THINKING_MODE_OURO_REPORT.md` | Branch B:Ouro 门控 + per-step 监督 + bug 修复 + 冻结补救 |

**脚本(`reproduce/`)**

| 文件 | 作用 |
|---|---|
| `run_thinking_probe.py` | Phase 0 诊断探针(training-free 扫描 + drift) |
| `run_finetune_coconut.py` | Branch A:COCONUT 课程微调(3 seeds × 5 数据集) |
| `run_finetune_ouro.py` | Branch B:Ouro 门控 + per-step 监督(已含 bug 修复) |
| `run_finetune_ouro_frozen.py` | Branch B 补救:冻结编码器 + 早停 |

**源码改动**

| 文件 | 改动 |
|---|---|
| `src/tabpfn/architectures/tabpfn_v2_5.py` | forward 加运行时可控 thinking 循环(coconut/ouro 双模式 + 可学习门控 + per-step logits 收集)。默认 `n_steps=1` 字节级不变,仅 plain 推理路径激活 |

**分支**:`thinking/coconut`、`thinking/ouro` 保留作开发历史;两者已合并回 `main`。

---

## 边界与诚实声明

- 5 个数据集偏小、偏易(单遍 acc 0.87–1.0),"近饱和"结论对**更难/更大**数据集未必成立 ——
  深度循环的潜在红利可能只在远未饱和的任务上才显现。这是最大的外推风险。
- 全程 n_estimators=1(便于干净测 thinking);集成下未测。
- 冻结实验证明了"损害来自编码器侵蚀",但未穷尽补救手段(KL 锚定、更大数据等);
  结论是"无益",非"原理上不可能"。
- 所有改动对默认推理零影响(`n_steps=1` 字节级一致),不影响 TabPFN 正常使用。

