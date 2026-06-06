# MiniLLM 项目学习路线

该项目是一个**从零实现 GPT 风格解码器专用 Transformer 语言模型**的教学项目，覆盖完整的大模型开发生命周期。推荐学习时长约 **4-6 周**（每天 2-3 小时）。

---

## 第一阶段：基础概念与数据准备（约 3 天）

在深入代码之前，先理解"为什么需要这些数据"。

| 步骤 | 内容 | 关键文件 |
|------|------|----------|
| 1.1 | 通读 `DESIGN.md`，了解项目整体设计哲学和架构决策 | `DESIGN.md` |
| 1.2 | 理解数据下载与清洗流程（WikiText-103 + TinyStories） | `src/phase1_data/download.py` |
| 1.3 | 掌握数据预处理（HTML 去除、文本规范化） | `src/phase1_data/preprocess.py` |
| 1.4 | 理解 Dataset/Dataloader 设计（numpy memmap 原理） | `src/phase1_data/dataset.py` |

**目标检查：** 能解释 memmap 为什么适合大规模语料，以及 next-token prediction 中 labels 如何从 input_ids 构造。

---

## 第二阶段：分词器原理（约 2 天）

| 步骤 | 内容 | 关键文件 |
|------|------|----------|
| 2.1 | Byte-Level BPE 算法原理 | `src/phase2_tokenizer/train_tokenizer.py` |
| 2.2 | 特殊 token 设计（PAD/UNK/BOS/EOS）与 post-processor | 同上 |
| 2.3 | 语料分词与序列打包策略（定长截断 + EOS 分隔） | `src/phase2_tokenizer/tokenize_data.py` |

**目标检查：** 能说出 BPE 的训练过程（初始化→计数→合并→迭代），以及为什么选择 Byte-Level 而非 Word-Level。

---

## 第三阶段：模型核心架构（约 7 天）⚠️ 核心阶段

这是整个项目最重要的部分，必须逐文件精读。

| 步骤 | 内容 | 关键文件 | 核心概念 |
|------|------|----------|----------|
| 3.1 | RoPE 旋转位置编码 | `src/phase3_model/rotary.py` | GPT-NeoX 配对旋转，频率预计算 |
| 3.2 | 因果自注意力机制 | `src/phase3_model/attention.py` | 合并 QKV 投影、因果掩码、手动计算 attention |
| 3.3 | 前馈网络（GELU/SiLU/SwiGLU） | `src/phase3_model/feedforward.py` | SwiGLU 的门控机制 |
| 3.4 | Transformer Block（Pre-LN 架构） | `src/phase3_model/transformer_block.py` | Pre-LN vs Post-LN 的差异和优劣 |
| 3.5 | 完整模型组装 + 权重绑定 | `src/phase3_model/model.py` | TiedLinear、GPT-2 初始化、generate 方法 |
| 3.6 | 配置系统与参数规模关系 | `src/common/config.py` | 三档模型配置（10M/25M/50M） |
| 3.7 | 运行单元测试验证理解 | `tests/test_model.py` | 9 个测试覆盖全部组件 |

### 关键公式

```
Attention(Q,K,V) = softmax(Q·K^T / sqrt(d_head) + mask) · V

参数量 ≈ n_layers × (4 × d_model² + 2 × d_model × d_ff) + vocab_size × d_model
```

### 目标检查

- 能手绘 Transformer 的计算图（输入 → Embedding → Block × N → LN → LM Head → 输出）
- 能解释 RoPE 为什么比绝对位置编码更好（相对位置关系内嵌于旋转）
- 能解释 Pre-LN 为什么训练更稳定（残差路径上梯度直接回传）
- 能说清 weight tying 如何节省 ~40% 参数量

---

## 第四阶段：预训练（约 4 天）

| 步骤 | 内容 | 关键文件 | 核心概念 |
|------|------|----------|----------|
| 4.1 | 训练循环全貌 | `src/phase4_pretrain/trainer.py` | 自回归语言建模的目标函数 |
| 4.2 | 学习率调度 | 同上 | Warmup → Cosine Annealing |
| 4.3 | 混合精度训练（AMP） | 同上 | CPU bf16 / CUDA fp16 |
| 4.4 | 梯度累积与裁剪 | 同上 | 模拟大 batch size |
| 4.5 | 训练指标与评估 | `src/phase4_pretrain/metrics.py` | PPL、梯度范数 |
| 4.6 | 检查点保存与恢复 | `src/phase4_pretrain/checkpoint.py` | 断点续训机制 |

**目标检查：** 能解释 `loss = -log P(next_token | previous_tokens)` 的数学含义，以及 PPL 与 loss 的关系（PPL = e^loss）。

---

## 第五阶段：监督微调（约 5 天）

| 步骤 | 内容 | 关键文件 | 核心概念 |
|------|------|----------|----------|
| 5.1 | SFT 数据准备与质量控制 | `src/phase5_sft/data_prepare.py` | Alpaca 数据集，去重与筛选 |
| 5.2 | 提示词模板设计 | `src/phase5_sft/templates.py` | Alpaca / ChatML / LLaMA 三种格式 |
| 5.3 | **Loss Masking（核心难点）** | `src/phase5_sft/loss_mask.py` | 仅对回复部分计算损失 |
| 5.4 | SFT 数据集构造 | `src/phase5_sft/sft_dataset.py` | 变长序列处理 |
| 5.5 | SFT 训练循环 | `src/phase5_sft/trainer.py` | Early Stopping、过拟合监控 |
| 5.6 | 模型评估与 Bad Case 分析 | `src/phase5_sft/evaluate.py` | 重复度检测、空输出检测 |
| 5.7 | 迭代优化工具 | `src/phase5_sft/iteration.py` | 问题分类、数据多样性分析 |

**目标检查：** 能解释为什么 SFT 中的 loss masking 至关重要（如果不 mask，模型学会的是复述 instruction 而非生成答案），以及如何使用 evaluation 指标指导数据迭代。

---

## 第六阶段：推理部署（约 3 天）

| 步骤 | 内容 | 关键文件 | 核心概念 |
|------|------|----------|----------|
| 6.1 | 文本生成算法 | `src/phase6_inference/generate.py` | Temperature / Top-K / Top-P / Repetition Penalty |
| 6.2 | 多轮对话管理 | 同上 | 历史截断策略 |
| 6.3 | 推理性能基准测试 | `src/phase6_inference/benchmark.py` | tokens/sec、延迟、内存 |
| 6.4 | Gradio Web 界面 | `src/phase6_inference/app.py` | 交互式聊天 |

**目标检查：** 能解释各种采样策略的区别——greedy（确定性但重复）、temperature（控制随机性）、top-k（截断低概率）、top-p（动态截断）。

---

## 第七阶段：实战运行（约 4 天）

按顺序执行完整的训练管线：

```bash
# 1. 数据准备
bash scripts/run_phase1_data.sh

# 2. 分词器训练
bash scripts/run_phase2_tokenizer.sh

# 3. 预训练
bash scripts/run_phase4_pretrain.sh

# 4. 监督微调
bash scripts/run_phase5_sft.sh

# 5. 启动聊天
bash scripts/run_phase6_demo.sh
```

---

## 学习重点排序

按重要性从高到低排列各模块：

| 优先级 | 模块 | 理由 |
|--------|------|------|
| 🔴 必学 | Phase 3（模型架构） | 整个项目的心脏，必须逐行理解 |
| 🔴 必学 | Phase 5（SFT + Loss Masking） | 工业级 SFT 的核心技巧 |
| 🟡 重要 | Phase 4（预训练） | 理解训练循环的工程细节 |
| 🟡 重要 | Phase 6（推理生成） | 采样策略直接影响用户体验 |
| 🟢 了解 | Phase 1（数据）、Phase 2（分词器） | 工程必备但原理相对独立 |

---

## 建议的学习方法

1. **先读配置再读代码：** 每个模块从 YAML 配置入手，理解超参数含义后再看实现
2. **画图辅助理解：** Transformer 的 tensor shape 变化（`(B, S, d_model)` → `(B, H, S, head_dim)` → ...）建议手绘维度流转图
3. **修改实验：** 修改 RoPE 为绝对位置编码，观察效果变化；修改 Pre-LN 为 Post-LN，对比训练稳定性
4. **从测试反推：** `tests/` 目录下的测试用例是最佳的使用示例，可辅助理解模块接口
