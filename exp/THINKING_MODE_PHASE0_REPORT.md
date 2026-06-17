# TabPFN Thinking Mode — Phase 0 诊断探针报告

生成时间: 2026-06-17
实验状态: ✅ 完成 (诊断性,非追求免费收益)
脚本: `reproduce/run_thinking_probe.py`, `reproduce/thinking_probe_results.log`

---

## 目标

把 COCONUT(潜空间反馈)与 Ouro(权重共享循环)的"计算换精度"思路接到 TabPFN v2.5。
v2.5 自带的 `AddThinkingRows`(64 个可学习 row token)恰好等价于 COCONUT 的 latent
thought 槽位,train+test 数据行等价于固定的"question 上下文"。在 forward 的块循环外
套一层运行时可控的循环(默认 `n_steps=1` 行为零改动),探针只为回答三件事:

1. 循环能否安全运行、`n_steps=1` 是否字节级一致?
2. 隐态漂移 `‖h^t − h^{t-1}‖` 随步数收敛(趋向不动点)还是发散/坍缩?
3. 准确率/NLL 是否随步数出现任何单调信号?

两种递归模式:
- **coconut**: 每遍仅把 thinking rows 输出隐态喂回,数据行复位到初始 embedding(latent-only)。
- **ouro**: 保留整个隐态,全栈重新循环。

---

## 结果

模型: `tabpfn-v2.5-classifier-v2.5_default.ckpt` | 设备: cuda | n_estimators=1

| 数据集 | 模式 | acc@1 | acc@4 | acc@8 | 趋势 |
|--------|------|-------|-------|-------|------|
| wine | coconut | 1.000 | 0.981 | 0.907 | 持平基线附近波动 |
| wine | ouro | 1.000 | **0.278** | 0.278 | 坍缩并冻结 |
| breast_cancer | coconut | 0.971 | 0.924 | 0.953 | 持平基线附近波动 |
| breast_cancer | ouro | 0.971 | **0.374** | 0.374 | 坍缩并冻结 |
| digits | coconut | 0.983 | 0.983 | 0.983 | 持平 |
| digits | ouro | 0.983 | **0.096** | 0.096 | 坍缩并冻结 |

**think drift 曲线(digits 为例)**:
- coconut: 812 → 361 → 189 → 197 → 173 → 168 → 187 → 211 (衰减后**震荡在有限区间**)
- ouro: 812 → 359 → 315 → **0.30 → 0.31 → 0.32**(第 4 步**塌到 ~0,锁死**)

---

## 关键发现

1. **行为不变验证通过**:三数据集 `steps=1` 全部 `dAcc=+0.0000`,与默认逐位一致(maxdiff 0.0)。

2. **Training-free 不白给收益**:无任何单调上升。固定深度训练的权重不是迭代算子,
   直接循环拿不到 Ouro 曲线 —— 与调研报告 §4 结论一致。

3. **两模式分道扬镳(最有价值的发现)**:
   - **ouro 全栈循环坍缩到平凡不动点**:drift 在第 4 步骤降到 ~0.2 并锁死,
     隐态被反复重整后收敛到与输入几乎无关的退化定点,预测冻死。
   - **coconut latent-only 反馈抗坍缩**:数据行每遍复位,只有 64 个 latent 槽位
     携带状态,drift 稳定在有限区间 —— 在"持续计算"而非"塌死"。

---

## 对 Phase 1(looped fine-tune)的含义

1. **走 coconut 路线,放弃裸 ouro 全栈循环**(数据上必然坍缩)。
2. **微调几乎必须加稳定化组件**:coconut drift 不收敛(震荡)说明权重距"好的迭代
   算子"有距离 —— 需要 Ouro 的可学习 per-step 残差门控标量,或 per-step 输出监督 + 熵正则。
3. **收益预期要诚实**:小数据集单遍已近饱和(acc 0.97–1.0),深度循环天花板有限,
   收益应放在更难数据集 / 更小训练集场景验证。

---

## 改动位置

`src/tabpfn/architectures/tabpfn_v2_5.py` — forward 块循环处加入运行时可控的
thinking 循环(读 `self._thinking_steps` / `self._thinking_mode`,默认关闭,仅在
plain 推理路径激活,cache/kv-build/checkpoint 路径不动)。
