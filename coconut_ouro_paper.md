# COCONUT 与 Ouro：开源现状、模型权重及 Training-Free 深度对比报告

> **调研日期**：2026 年 6 月 17 日
> **调研范围**：代码开源情况、预训练模型权重可用性、是否支持 Training-Free 推理时扩展

---

## 1. 论文基本信息

| 项目 | COCONUT | Ouro |
|---|---|---|
| **全称** | Chain of Continuous Thought | Scaling Latent Reasoning via Looped Language Models |
| **arXiv** | [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) | [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) |
| **发表状态** | COLM 2025（已接收） | arXiv 预印本（v4，2025 年 11 月） |
| **机构** | FAIR at Meta / UC San Diego | ByteDance Seed、UC Santa Cruz、Princeton 等 |
| **核心作者** | Shibo Hao, Sainbayar Sukhbaatar, Yuandong Tian 等 | Rui-Jie Zhu, Zixuan Wang 等 |
| **核心机制** | 将 LLM 最后一层隐状态作为"连续思维"直接回馈输入，在潜空间中完成推理 | 参数共享的循环 Transformer，同一组权重在推理时迭代执行多次 |

---

## 2. 代码开源情况

### 2.1 COCONUT

**官方仓库**：[facebookresearch/coconut](https://github.com/facebookresearch/coconut)（MIT 许可证，⭐ 1.6k）

COCONUT 提供了**完整的官方实现代码**，仓库结构清晰，包含以下核心文件：

| 文件/目录 | 说明 |
|---|---|
| `coconut.py` | 核心模型实现，定义 `Coconut` 类 |
| `run.py` | 训练与评估入口 |
| `dataset.py` | 数据集加载与处理 |
| `args/*.yaml` | 各任务（GSM8k、ProntoQA、ProsQA）的完整训练配置 |
| `preprocessing/` | 数据预处理脚本 |
| `requirements.txt` | 依赖列表 |

**代码特点**：
- 实验级代码，依赖 [wandb](https://wandb.ai/) 进行日志记录；
- 支持多 GPU 训练（`torchrun`），论文实验基于 4×A100（80GB）；
- 无 vLLM / SGLang 等推理框架集成。

### 2.2 Ouro

**官方仓库状态**：论文主页 [ouro-llm.github.io](https://ouro-llm.github.io/) 标注为 **"Code (Coming Soon)"**，截至调研日期，官方训练代码**尚未正式开放**。

**第三方社区复现**：[rkstgr/LoopLM](https://github.com/rkstgr/LoopLM)（非官方，⭐ 5）

该复现版本实现了 Ouro 的核心架构，包括：
- 参数共享循环机制（Shared-weight Recurrence）
- Sandwich Normalization
- 早退门控（Exit Gate）
- 多步损失（Multi-step Loss）
- 支持 `small`（~100M）、`ouro_1_4b`（1.4B）、`ouro_2_6b`（2.6B）三种配置

**推理框架集成（官方支持）**：尽管训练代码未开放，Ouro 的推理侧已完成工业级集成：
- **vLLM**：PR 已合并，可直接 `vllm serve "ByteDance/Ouro-1.4B"` 部署；
- **SGLang**：同样已集成，支持 OpenAI 兼容 API。

### 2.3 代码开源对比小结

| 维度 | COCONUT | Ouro |
|---|---|---|
| **官方代码** | 完整开源（MIT） | 训练代码未开源 |
| **推理框架集成** | 无 | vLLM / SGLang 均已集成 |
| **第三方复现** | [OpenCoconut](https://github.com/casper-hansen/OpenCoconut) 等多个 | [rkstgr/LoopLM](https://github.com/rkstgr/LoopLM) |
| **代码成熟度** | 学术实验级 | 推理侧工业级，训练侧暂缺 |

---

## 3. 预训练模型权重

### 3.1 COCONUT

**官方权重**：**至今未发布。**

论文发布时声明：

> "We are releasing our training recipe and data, and **intend to release the model weights in the future**."

该承诺截至目前仍未兑现。

**第三方权重**：[tsrigo/coconut](https://huggingface.co/tsrigo/coconut)（HuggingFace，非官方）

- 由社区用户在单张 A100 40GB 上自行复现训练；
- 仅包含 GSM8k 任务的 checkpoint（`save_models/gsm-coconut/checkpoint_22`）；
- 基础模型为 GPT-2（`openai-community/gpt2`，117M 参数），规模极小；
- 非官方，质量和可靠性无保障。

**使用示例（第三方权重）**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from coconut import Coconut
import torch

model_id = "openai-community/gpt2"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 添加特殊 latent tokens
tokenizer.add_tokens("<|start-latent|>")
tokenizer.add_tokens("<|end-latent|>")
tokenizer.add_tokens("<|latent|>")

# 加载第三方 checkpoint
saved_weights = torch.load("checkpoint_22", map_location="cuda")
model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
model.load_state_dict(saved_weights, strict=False)
```

### 3.2 Ouro

**官方权重**：**已发布 4 个模型**，全部托管于 HuggingFace [ByteDance 组织](https://huggingface.co/ByteDance)，**Apache-2.0 许可证**。

| 模型 | 参数量 | 用途 | 下载量（近期） |
|---|---|---|---|
| [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) | 1.4B | 通用基础模型 | ~48,000 |
| [ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B) | 2.6B | 通用基础模型（更强） | — |
| [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) | 1.4B | 推理专用（SFT 后） | ~8,975 |
| [ByteDance/Ouro-2.6B-Thinking](https://huggingface.co/ByteDance/Ouro-2.6B-Thinking) | 2.6B | 推理专用（SFT 后） | — |

**模型架构参数（以 Ouro-1.4B 为例）**：

| 配置项 | 值 |
|---|---|
| 参数量 | 1.4B |
| Transformer 层数 | 24 |
| 循环步数（默认） | 4（`total_ut_steps=4`） |
| 隐层维度 | 2048 |
| 注意力机制 | Multi-Head Attention (MHA) |
| FFN 激活函数 | SwiGLU |
| 位置编码 | RoPE |
| 词表大小 | 49,152 |
| 上下文长度 | 4K（预训练），可扩展至 64K |
| 归一化方式 | Sandwich RMSNorm |
| 预训练数据量 | 7.7T tokens |

**快速使用**：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ByteDance/Ouro-1.4B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> **注意**：需使用 `transformers < 4.56.0`，推荐 `transformers==4.54.1`，以避免兼容性问题。

### 3.3 模型权重对比小结

| 维度 | COCONUT | Ouro |
|---|---|---|
| **官方权重** | **未发布** | **已发布**（4 个模型，Apache-2.0） |
| **第三方权重** | 有（GPT-2 规模，非官方） | 无需，官方已足够 |
| **基础模型规模** | GPT-2（117M） | 1.4B / 2.6B（从头预训练） |
| **可直接使用** | 否（需自行训练） | **是** |

---

## 4. 是否 Training-Free？

这是两者最本质的区别，需要从**方法论层面**仔细辨析。

### 4.1 COCONUT：完全不是 Training-Free

COCONUT 的核心机制要求对模型进行专门的**多阶段课程训练（Multi-stage Curriculum Training）**，训练流程如下：

```
阶段 0：用标准 CoT 数据对 GPT-2 进行监督微调（Stage 0 CoT Training）
    ↓
阶段 1：将推理链最后 1 步替换为 c 个 latent tokens，继续训练
    ↓
阶段 2：将最后 2 步替换为 latent tokens
    ↓
    ……（逐阶段递增）
    ↓
阶段 k：全部推理步骤均在潜空间中完成（纯 latent reasoning）
```

**为什么不能 Training-Free？**

1. **模型必须通过训练才能"学会"在潜空间编码推理状态**：连续思维向量（continuous thought）是梯度下降的产物，不是现有 LLM 天然具备的能力；
2. **对任意标准 LLM 无法零样本迁移**：无法将 GPT-4、Llama-3 等直接转换为 COCONUT 模式；
3. **推理时的 latent thought 数量（`c_thought`）必须与训练配置一致**：训练时设定 `c=2`，推理时不能随意改为 `c=8`。

> **一句话总结**：COCONUT 是一种**训练范式（Training Paradigm）**，而非推理技巧。它从根本上改变了模型的训练目标和数据格式，推理时的"计算换精度"能力是训练出来的，而非即插即用的。

**附：Training-Free 的衍生方法 NoisyCoconut**

受 COCONUT 启发，TMLR 在投论文 **NoisyCoconut**（arXiv 2026）提出了一种真正 Training-Free 的方法：在推理时向 LLM 的隐层注入受控噪声，生成多条多样化推理路径，通过路径间的一致性作为置信度信号。实验表明，在数学推理任务上可将错误率从 40–70% 降低至 15% 以下。但这是独立于 COCONUT 的新方法，并非 COCONUT 本身。

### 4.2 Ouro：推理时调参是 Training-Free，但获得模型需预训练

Ouro 的情况更为微妙，需要区分两个层面：

#### 层面 A：对已有 Ouro 模型——循环步数调整是 Training-Free 的 ✅

Ouro 的 HuggingFace 模型卡明确支持在推理时**无需任何训练**地调整循环步数：

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("ByteDance/Ouro-1.4B")

# 默认 4 步，可自由调整以换取更高精度（training-free）
config.total_ut_steps = 6

# 早退阈值：1.0 = 始终执行全部步数；降低该值可加速推理
config.early_exit_threshold = 0.8

model = AutoModelForCausalLM.from_pretrained(
    "ByteDance/Ouro-1.4B",
    config=config,
    device_map="auto"
)
```

这正是"计算换精度"的直接体现：**增加 `total_ut_steps` → 更多潜空间迭代 → 更高精度，无需任何训练**。

#### 层面 B：获得 Ouro 模型本身——需要大规模预训练 ❌

- Ouro 架构（参数共享的循环 Transformer）必须在**预训练阶段**就嵌入，配合熵正则化目标（entropy-regularized objective）联合训练；
- 不存在"将 Llama-3 或 Qwen 改造为 Ouro"的 training-free 路径；
- 官方训练流程共 7 个阶段，消耗 7.7T tokens，计算成本极高。

> **一句话总结**：Ouro 的**推理时扩展（调整循环步数）是 Training-Free 的**；但获得一个 Ouro 模型本身需要大规模预训练。好消息是官方已发布权重，用户可以**直接跳过训练阶段**，立即享受 training-free 的推理时计算扩展。

### 4.3 Training-Free 对比小结

| 维度 | COCONUT | Ouro |
|---|---|---|
| **推理时增加计算量** | 需对应训练配置，**不可随意调整** | **直接修改 `total_ut_steps`，无需训练** |
| **对现有 LLM 的适用性** | 需从头走完多阶段课程训练，**不可迁移** | 需使用官方 Ouro 权重，**不可从现有 LLM 转换** |
| **零样本迁移** | **不支持** | **不支持**（需 Ouro 架构） |
| **实际使用门槛** | 高（需 4×A100，自行训练） | **低**（直接 `from_pretrained` 加载即用） |

---

## 5. 综合对比总览

| 对比维度 | COCONUT | Ouro |
|---|---|---|
| **代码开源** | 完整开源（MIT） | 训练代码未开源；推理已集成 vLLM/SGLang |
| **官方模型权重** | **未发布** | **已发布**（1.4B / 2.6B，Apache-2.0） |
| **第三方权重** | 有（GPT-2 规模，非官方） | 无需，官方已足够 |
| **是否 Training-Free** | **否**，必须完整走多阶段课程训练 | **推理时调参是 Training-Free**；获得模型本身需预训练 |
| **推理时"计算换精度"** | 通过增加 latent thought 数量，但受训练配置约束 | **直接支持**：修改 `total_ut_steps` 即可 |
| **对现有 LLM 的适用性** | 需从头微调，不可直接迁移 | 需使用官方 Ouro 权重，不可从现有 LLM 转换 |
| **实用门槛** | 高（需 GPU 资源自行训练） | **低**（直接下载使用） |
| **适合场景** | 研究潜空间推理机制、复现论文实验 | 直接部署使用、探索推理时计算扩展 |
| **许可证** | MIT（代码）；权重未发布 | Apache-2.0（代码 + 权重） |

---

## 6. 对 TabPFN 等表格模型的应用启示

结合上述调研，若要在 TabPFN 等表格模型中实践"计算换精度"，两条路线的可操作性差异显著：

**Ouro 路线（更具可操作性）**

Ouro 推理时循环步数可调的机制是真正 training-free 的，可作为架构设计的直接参考。具体而言，可借鉴其"参数共享循环 Transformer + 熵正则化早退"的设计，在 TabPFN 中引入可配置的迭代深度。需注意，Ouro 的循环步数在预训练时固定为 4 步，超出此范围的外推效果尚不明确，需实验验证。

**COCONUT 路线（潜力更高，但成本更大）**

若要在 TabPFN 中引入潜空间推理，需要设计类似的多阶段课程训练，工程成本较高。但对于表格数值数据而言，潜空间推理的优势更为显著——避免了文本空间的离散化误差，连续的特征精炼过程与表格数据的天然连续性高度契合。

---

## 参考资料

| 资源 | 链接 |
|---|---|
| COCONUT 论文 | [arXiv:2412.06769](https://arxiv.org/abs/2412.06769) |
| COCONUT 官方代码 | [facebookresearch/coconut](https://github.com/facebookresearch/coconut) |
| COCONUT 第三方权重 | [tsrigo/coconut](https://huggingface.co/tsrigo/coconut) |
| Ouro 论文 | [arXiv:2510.25741](https://arxiv.org/abs/2510.25741) |
| Ouro 项目主页 | [ouro-llm.github.io](https://ouro-llm.github.io/) |
| Ouro-1.4B 权重 | [ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) |
| Ouro-1.4B-Thinking 权重 | [ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking) |
| Ouro 第三方复现 | [rkstgr/LoopLM](https://github.com/rkstgr/LoopLM) |
| NoisyCoconut（Training-Free 衍生） | [OpenReview](https://openreview.net/forum?id=5aatZPiCv8) |


COCONUT 追求的是两件事同时成立：
让推理过程从离散文本转到连续潜空间里进行；
在更少显式文本 token 的情况下，尽量保留甚至提升推理能力。
所以它不是简单地把推理链“缩短”了，而是把原来一部分文本推理步骤，替换成了continuous thoughts（连续潜变量思考）。论文里明确报告：在 GSM8k 上，COCONUT 相对 No-CoT 有显著提升；并且随着连续 thought 数量增加，性能还会继续上升。这说明它不只是“少输出几个 token”，而是真的把额外计算用于推理了。
但如果和完整的语言 CoT比，就不能简单说“全面更强”。论文中的结论更细：
在 ProntoQA、ProsQA 这类任务上，COCONUT 能够用更少生成 token 获得更好的效果；
在 GSM8k 上，它没有超过完整 CoT，但在“精度 / 推理长度”这个权衡上更优。

提升是真实的，且单调可控
Ouro 最核心的卖点是：随着循环步数（total_ut_steps）增加，性能单调上升。这在论文里被称为 monotonic improvement with compute，是区别于普通 Transformer 直接循环的关键。
循环步数
性能趋势
1 步（相当于普通前向）
基线
2 步
提升
4 步（训练时最大值）
最优
>4 步（超出训练范围）
提升趋于饱和，不保证单调
在哪类任务上提升最明显
Ouro 的增益主要集中在推理密集型任务上，例如：
数学推理（MATH、GSM8k）
代码生成（HumanEval）
逻辑推理
在这些任务上，相比同等参数量的标准 Transformer，Ouro 通过 4 步循环能获得相当于参数量翻倍甚至更多的性能增益，同时推理时显存占用远低于直接扩大模型。
但有一个重要前提
Ouro 的单调提升依赖于专门的训练方式：
权重共享的循环块在训练时就以 T_max=4 步循环训练；
每一步都有独立的 LM head 监督（per-step loss）；
有熵正则防止模型"偷懒"只靠最后一步。
