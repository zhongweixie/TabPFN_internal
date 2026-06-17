# TabPFN Thinking Mode — Branch A (COCONUT) 报告

生成时间: 2026-06-17
分支: `thinking/coconut`
状态: ✅ 完成首轮
脚本: `reproduce/run_finetune_coconut.py`, `reproduce/finetune_coconut_results.log`

---

## 方法(忠实于 COCONUT,无门控)

- **循环 = latent-only 反馈**:v2.5 的 `AddThinkingRows`(64 个可学习 token)充当
  COCONUT 的 continuous-thought 槽位;train+test 数据行是固定"question 上下文",
  每遍复位到初始 embedding。只有 thinking rows 跨遍携带状态。
- **课程训练**(COCONUT 灵魂):把"逐步用 latent 替换文本推理步"映射为"逐步增加
  latent 循环遍数":epoch 0–3 → 1 遍,4–7 → 2 遍,8–11 → 4 遍。
- **监督**:仅最后一遍 query 的 cross-entropy。
- **可训练**:全模型,含 `add_thinking_rows.row_token_values_TE`(已验证收到梯度并更新)。
- **训练 forward 路径**:微调的 batched 引擎把 `cat([X_ctx,X_qry])` 全表喂给
  `model.forward`(plain 非 cache 路径),thinking 循环在训练时触发、梯度穿过循环反传。

实现要点:自定义紧凑训练循环(复用 `get_preprocessed_dataset_chunks` +
`fit_from_preprocessed` + `forward`),课程调度与多深度评估完全自控。

---

## 结果(pre = 微调前, post = 课程微调后;同一推理深度对比)

模型 `tabpfn-v2.5-classifier-v2.5_default.ckpt` | n_estimators=1 | 12 epochs | lr 1e-5

### wine (178×13, 3类) — 循环被驯化收敛 ✅

| steps | pre acc / nll | post acc / nll | post drift |
|------|---------------|----------------|------------|
| 1 | 1.000 / 0.0076 | 1.000 / 0.0049 | — |
| 2 | 0.981 / 0.1008 | **1.000 / 0.0116** | 413 → 117 |
| 4 | 0.981 / 0.0528 | **1.000 / 0.0110** | 413 → 117 → **6.1 → 1.0** |
| 8 | 0.907 / 0.2251 | **1.000 / 0.0110** | … → 0.4 → 0.3 → 0.4 → 0.3 |

pre 时深度循环掉点(steps=8 → 0.907),post 全部 1.000。**drift 从震荡(100–160)
变为收敛到不动点(→0.3)** —— 课程微调把循环训成了稳定迭代算子,且未用门控。

### breast_cancer (569×30, 2类) — 生效但深度收益不明确

| steps | pre acc / nll | post acc / nll |
|------|---------------|----------------|
| 1 | 0.971 / 0.0868 | 0.959 / 0.0934 |
| 4 | 0.924 / 0.1905 | 0.953 / 0.1567 |
| 8 | 0.953 / 0.1478 | 0.936 / 0.1616 |

drift 仍在 250–350 震荡,未收敛。深度循环小幅恢复但噪声大。

### digits (1797×64, 10类) — 生效但深度仍掉点

| steps | pre acc / nll | post acc / nll |
|------|---------------|----------------|
| 1 | 0.983 / 0.0447 | **0.985 / 0.0475** |
| 2 | 0.970 / 0.0945 | 0.974 / 0.0937 |
| 4 | 0.983 / 0.0625 | 0.965 / 0.1148 |
| 8 | 0.983 / 0.0666 | 0.957 / 0.1258 |

steps=1 微升,steps≥4 反而略降,drift 仍震荡(200–300)未收敛。

---

## 结论

1. **COCONUT 课程能把循环训成稳定迭代算子**(wine: drift 震荡 → 收敛到 ~0.3,
   且 deep inference 不再掉点) —— 验证了 Phase 0 "需要稳定化才能收敛"的判断,
   **且无需门控**,仅靠课程 + 最终步监督即可。

2. **但纯最终步监督在更难数据上不足**:breast_cancer / digits 的深度循环 drift 未
   收敛、深度收益不明确甚至掉点。

3. **wine 收敛需打折看待**:124 训练行、单 batch、acc 本就 1.0,drift 收敛可能部分
   是"模型学会让循环趋于恒等"而非"学会用深度推理"。

4. **直接指向 Branch B(Ouro)要补的**:per-step 输出监督(每遍 decode 都算 loss)
   会强迫每一遍都有用,而非只让循环安静下来 —— 这正是更难数据上让"深度=更好"
   所缺的训练信号。

---

## 局限与下一步

- 单 batch / 小数据集,信号弱;应在更大、更难的 OpenML 数据集上复跑。
- 课程仅 3 段、12 epoch;可拉长课程、加 epochs。
- 未做多 seed 重复,wine 的 1.000 需多次确认。
- 评估前需用 `state_dict` 快照恢复微调权重(`est.fit` 会从 ckpt 重载,曾导致 pre==post 的 bug,已修)。
