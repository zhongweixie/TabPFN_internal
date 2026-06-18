# TabPFN Thinking Mode — Branch A (COCONUT) 报告

生成时间: 2026-06-17(2026-06-18 经审查修正:框架去夸大 + 多 seed 强化统计)
分支: `thinking/coconut`
状态: ✅ 完成(已修正)
脚本: `reproduce/run_finetune_coconut.py`, `reproduce/finetune_coconut_results.log`

---

## 方法(COCONUT 启发,非复刻;无门控)

> **命名澄清(经审查修正)**:此分支机制上是"**只让 latent 行携带状态的权重共享
> 深度递归**",受 COCONUT *启发*但**不是忠实复刻**。三处关键差异需明示:
> 1. **槽位语义**:COCONUT 连续思维是序列自回归(位置 t→t+1);这里 64 个槽位并行
>    存在、被全栈重复 N 遍 —— 更接近"latent-only 的 Ouro 递归"。
> 2. **课程**:COCONUT 课程增加 *latent token 数*(替换更多文本步);这里增加 *循环遍数*。
>    是"深度"而非"宽度",非严格同构。
> 3. **监督**:COCONUT 通过分阶段课程获得隐式 per-step 信号;这里只有最终遍 query 的
>    cross-entropy,无 per-step 监督(这正是 Branch B/Ouro 要补的)。

- **循环 = latent-only 反馈**:v2.5 的 `AddThinkingRows`(64 个可学习 token)被借用为
  latent 槽位;train+test 数据行是固定"question 上下文",每遍复位到初始 embedding。
  只有 thinking rows 跨遍携带状态。
- **课程训练**(借鉴 COCONUT 分阶段思想):逐步增加 latent 循环遍数:
  epoch 0–3 → 1 遍,4–7 → 2 遍,8–11 → 4 遍。
- **监督**:仅最后一遍 query 的 cross-entropy(无 per-step 监督)。
- **可训练**:全模型,含 `add_thinking_rows.row_token_values_TE`。已实测梯度流:
  250/250 参数收到梯度(thinking rows 1.46、block 3.69、feature embedder 9.09)。
  数据行第 2 遍起以 `.detach()` 重注入,故循环内不对固定 context 二次求导 —— 这是
  "数据行=固定 context"的正确语义,非 bug。
- **训练 forward 路径**:微调的 batched 引擎把 `cat([X_ctx,X_qry])` 全表喂给
  `model.forward`(plain 非 cache 路径),thinking 循环在训练时触发、梯度穿过循环反传。
- **评估 forward 路径(经实测确认)**:`fit_preprocessors` 下 `predict_proba` 是
  **单次无 cache forward**,thinking 守卫满足、循环在预测时**确实激活**(实测 steps=4
  vs steps=1 输出 maxdiff 0.74–0.80,drift 记录 4 步)。post 数字不是缓存伪影。

实现要点:自定义紧凑训练循环(复用 `get_preprocessed_dataset_chunks` +
`fit_from_preprocessed` + `forward`),课程调度与多深度评估完全自控。

---

## 结果(强化版:3 seeds × 5 数据集 × 多 batch/epoch)

模型 `tabpfn-v2.5-classifier-v2.5_default.ckpt` | n_estimators=1 | 12 epochs |
lr 1e-5 | 每 epoch ~3–6 个 batch(自适应 chunk)| 报告 `Δacc(post−pre)` mean±std。

> **方法强化说明**:初版单 seed、每 epoch 仅 1 个 batch(`max_data_size=None`),
> 审查指出统计无效。本版改为 3 seeds、自适应 chunk 使每 epoch 多个梯度步,并加入
> 两个更大/更难的 OpenML 数据集(phoneme 5404 行、qsar-biodeg 1055 行)。

`Δacc(post−pre)` 按 seed 配对求差,正 = 微调后该深度更好:

| 数据集 (base pre@1) | steps=1 | steps=2 | steps=4 | steps=8 |
|---|---|---|---|---|
| wine (1.000) | −0.006±0.009 | **+0.049±0.023** | +0.086±0.109 | **+0.093±0.030** |
| breast_cancer (0.971) | −0.006±0.008 | +0.004±0.006 | +0.010±0.007 | −0.006±0.005 |
| digits (0.985) | +0.001±0.003 | +0.013±0.011 | +0.007±0.012 | +0.013±0.020 |
| phoneme (0.906) | −0.006±0.005 | **+0.019±0.014** | +0.006±0.010 | +0.017±0.012 |
| qsar-biodeg (0.869) | +0.002±0.005 | −0.006±0.027 | **−0.028±0.042** | −0.020±0.039 |

读法:
- **wine / phoneme**:深度循环(steps≥2)post 一致优于 pre,std 小于均值 → 信号较实。
- **digits**:全为小正但都在 1σ 内,**不显著**。
- **breast_cancer**:steps=4 微正、steps=8 转负,**深度非单调、收益边际**。
- **qsar-biodeg**:深度循环 post **更差**(steps=4 −0.028±0.042),且 std 大 → 微调后
  深度反而有害,方差升高。

> 注:初版单 seed 曾报告 wine steps=8 "0.907→1.000"、drift "收敛到 0.3",看似强信号。
> 多 seed 下 wine 仍正(+0.093±0.030)但其余数据集**未复现这种干净收敛**,说明单 seed
> 的乐观印象有相当部分是噪声 / 数据集太易。

---

## 结论(多 seed 后修正)

1. **看不到稳健的"深度=更好"**。跨 5 数据集,深度循环的收益**不一致**:wine/phoneme
   小幅正且较稳,digits 不显著,breast_cancer 边际且非单调,qsar **明显变差**。纯最终步
   监督的 COCONUT 式课程**不足以**让"更多遍 = 更好"普遍成立。

2. **steps=1 普遍略降(−0.006 量级)**。微调让模型适配了"多遍"分布,反而轻微牺牲了
   单遍性能 —— 说明课程确实改变了权重,但没换来净增益。

3. **初版单 seed 结论被推翻**:wine "0.907→1.000 + drift 收敛到 0.3" 看似强,多 seed 下
   其余数据集均未复现这种干净收敛,大部分是噪声 / 数据集太易造成的乐观偏差。
   **这是 A(修框架)+ B(加统计)联合带来的最重要修正。**

4. **机制结论站得住**:代码正确性经实测确认(n_steps=1 字节级一致、训练/评估两条路径
   thinking 均激活、250/250 参数收到梯度);失败的是**方法**(最终步监督 + 深度课程),
   不是实现。

5. **直接指向 Branch B(Ouro)**:需要 **per-step 输出监督**(每遍 decode 都算 loss、按
   exit 概率加权 + 熵正则)强迫每一遍都有用,以及**可学习 per-step 残差门控**让模型在难
   数据上能"选择少算"而非被迫多算致害(qsar 的退化正是缺这个旋钮)。

---

## 局限与下一步

- 仍是 3 seed / 5 数据集 / 单 ckpt;结论是"未见稳健增益",非"证明无效"。
- 课程仅 3 段、12 epoch、lr 固定;未扫超参,可能存在更优配置。
- n_estimators=1(单 forward,便于干净测 thinking);集成下行为未测。
- **评估正确性**:已实测 `predict_proba` 走单次无 cache forward、thinking 在预测时
  激活(steps=4 vs 1 maxdiff 0.74–0.80);`est.fit` 会从 ckpt 重载权重,故用 `state_dict`
  快照恢复微调权重(初版此处有 pre==post 的 bug,已修并验证)。
