# 从零训练大语言模型 — 方案设计文档

> **目标:** 在 CPU 环境下，从零完成一个小规模大语言模型的全流程训练（数据采集 → 预训练 → 微调 → 推理），用于学习理解 LLM 的核心原理。
>
> **硬件约束:** CPU-only 训练 / 推理
>
> **设计原则:** 专业流程、小规模数据、小参数量、可复现、可学习

---

## 目录

1. [项目概览](#1-项目概览)
2. [技术栈选型](#2-技术栈选型)
3. [阶段一：数据工程](#3-阶段一数据工程)
4. [阶段二：分词器训练](#4-阶段二分词器训练)
5. [阶段三：模型架构设计](#5-阶段三模型架构设计)
6. [阶段四：预训练](#6-阶段四预训练)
7. [阶段五：监督微调（重点）](#7-阶段五监督微调重点)
8. [阶段六：推理与评估](#8-阶段六推理与评估)
9. [阶段七：模型冻结与保存](#9-阶段七模型冻结与保存)
10. [项目目录结构](#10-项目目录结构)
11. [里程碑与时间估算](#11-里程碑与时间估算)
12. [执行指南](#12-执行指南)
13. [附录](#附录)

---

## 1. 项目概览

### 1.1 核心设计思路

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 数据工程  │ →  │ 分词器    │ →  │ 预训练    │ →  │ 监督微调  │ →  │ 推理部署  │
│          │    │ 训练      │    │ (PT)     │    │ (SFT)    │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     ↓               ↓               ↓               ↓               ↓
 公开语料        BPE分词器      GPT-like      指令微调       文本生成
 清洗/分词       词汇表8k-16k   自回归LM       对话能力       模型冻结
```

### 1.2 规模参数

| 维度 | 数值 | 说明 |
|------|------|------|
| 模型参数量 | ~10M - 50M | 小规模，CPU 可训练 |
| Transformer 层数 | 4 - 8 | 浅层网络 |
| 注意力头数 | 4 - 8 | 标准配置 |
| 隐藏维度 | 256 - 512 | 较小嵌入维度 |
| 词汇表大小 | 8,000 - 16,000 | BPE tokenizer |
| 上下文长度 | 256 - 512 | 短序列，降低计算量 |
| 预训练数据量 | 100MB - 500MB | WikiText + 公开语料 |
| 微调数据量 | 5,000 - 20,000 条 | 指令数据 |

### 1.3 为什么这套参数可以在 CPU 上训练

以上述最大配置（50M 参数）估算：

- 模型内存占用：~200MB（fp32）/ ~100MB（fp16）
- 单步训练内存（batch_size=8, seq_len=512）：~500MB - 1GB
- 完全在 8GB+ RAM 的普通 PC 可运行
- 训练速度：约 500-2000 tokens/s（取决于 CPU 性能）

---

## 2. 技术栈选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 深度学习框架 | **PyTorch** | 生态成熟，CPU 支持好，学习资料丰富 |
| 分词库 | **HuggingFace Tokenizers** | Rust 实现，CPU 上极快训练 BPE |
| 预训练加速 | **PyTorch AMP** (可选) | BF16 混合精度，CPU 上减少内存 |
| 数据处理 | **HuggingFace Datasets** | 流式加载，内存高效 |
| 模型定义 | 纯 **PyTorch nn.Module** | 从零手写，深入理解每个组件 |
| 训练循环 | 纯 **PyTorch** 训练循环 | 不依赖 Trainer，理解每个步骤 |
| 推理服务 | **Gradio** | 轻量 Chat UI，比 Flask 更简单 |
| 实验追踪 | **TensorBoard** | 可视化 loss / lr / ppl 曲线 |
| 日志系统 | **Python logging + 文件输出** | 结构化日志，按阶段/日期分文件 |
| 配置管理 | **YAML / dataclass** | 清晰可维护 |

---

## 3. 阶段一：数据工程

### 3.1 数据来源

由于没有现成训练数据，使用公开可下载的数据集：

#### 预训练数据（无标签文本）

| 数据集 | 大小 | 语言 | 下载方式 |
|--------|------|------|----------|
| **WikiText-2** | ~2M tokens | 英文 | `torchtext.datasets.WikiText2` |
| **WikiText-103** | ~103M tokens | 英文 | `torchtext.datasets.WikiText103` |
| **BookCorpus** 子集 | ~100MB | 英文 | HuggingFace `bookcorpus` |
| **TinyStories** | ~80MB | 英文 | HuggingFace `roneneldan/TinyStories` |
| **中文维基百科** | ~200MB | 中文 | Wikimedia dumps |

**推荐组合方案（中英双语小型模型）：**
- TinyStories（英文简单文本）+ 中文维基子集 = ~100-200MB 原始文本

**推荐组合方案（纯英文最小可用）：**
- WikiText-103（英文百科）+ TinyStories（英文故事）= ~180MB 原始文本

#### 微调数据（指令-响应对）

| 数据集 | 大小 | 说明 |
|--------|------|------|
| **Alpaca (GPT-4 生成)** | 52K 条 | `tatsu-lab/alpaca` |
| **Dolly-15K** | 15K 条 | `databricks/databricks-dolly-15k` |
| **Belle** (中文) | ~500K+ 条 | `BelleGroup/train_0.5M_CN` |
| **self-instruct 子集** | 自定义 | 手动采样 5K-10K 条 |

**推荐：** Alpaca 子集（5K-10K 条），高质量英文指令数据

### 3.2 数据处理流程

```
原始文本 → 清洗 → 分句/分段 → 拼接/截断 → Tokenize → 保存为 .bin 文件
```

#### 3.2.1 文本清洗规则

```python
# 清洗 pipeline
def clean_text(text: str) -> str:
    # 1. 移除 HTML 标签
    # 2. 统一空白字符（多个空格→单个空格）
    # 3. 统一换行符
    # 4. 移除不可打印字符
    # 5. 可选：过滤过短行（< 10 字符）
    return cleaned
```

#### 3.2.2 文档分块策略

```python
# 将文档切分为固定长度的 token 序列
# 预训练时使用滑动窗口或拼接策略

# 策略 A：文档拼接 + 随机截断（推荐，减少 padding 浪费）
# 将多个文档用 <EOS> 拼接，达到 seq_len 后截断

# 策略 B：文档填充
# 每个文档独立，不足 seq_len 用 PAD 填充
```

---

## 4. 阶段二：分词器训练

### 4.1 Tokenizer 选择

选择 **BPE (Byte-Pair Encoding)** 作为分词算法：

- 平衡 OOV（未登录词）与词汇表大小
- GPT 系列使用的标准方案
- HuggingFace `tokenizers` 库在 CPU 上极快训练

### 4.2 词汇表规模

| 方案 | 词汇表大小 | 适用场景 |
|------|-----------|----------|
| 极简 | 4,096 | 超小模型，快速实验 |
| **推荐** | **8,192** | 小模型最佳平衡点 |
| 标准 | 16,384 | 较大模型 |

### 4.3 训练流程

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. 初始化 BPE 分词器（字节级，不引入外部预分词器依赖）
tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 2. 配置训练器（BPE 从 256 个字节开始自动学习合并规则）
trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
    min_frequency=2,  # Token 最小出现频率；出现 1 次的不纳入词汇表（去除噪声）
)

# 3. 在文本文件列表上训练
tokenizer.train(files=["data/corpus.txt"], trainer=trainer)

# 4. 保存分词器
tokenizer.save("models/tokenizer.json")
```

### 4.4 Tokenizer 输出格式

```python
# 编码示例
text = "The quick brown fox"
encoded = tokenizer.encode(text)
# → ids: [<BOS>, 1234, 567, 8901, 234, <EOS>]
# → tokens: ['<BOS>', 'The', 'Ġquick', 'Ġbrown', 'Ġfox', '<EOS>']
# 注意：BPE 的 Ġ 表示前导空格（GPT 风格）
```

---

## 5. 阶段三：模型架构设计

### 5.1 整体架构：Decoder-Only Transformer (GPT-like)

```
                    ┌──────────────┐
                    │   Output     │
                    │ (vocab_size) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  LM Head     │
                    │ (Linear)     │
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │   Final LayerNorm       │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  TransformerBlock × N   │
              │  ┌───────────────────┐  │
              │  │ Multi-Head Causal │  │
              │  │    Attention      │  │
              │  ├───────────────────┤  │
              │  │   Feed-Forward    │  │
              │  │   (SwiGLU/ReLU)   │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  Token Embedding        │
              │  + Position Embedding   │
              └────────────┬────────────┘
                           │
                    ┌──────▼───────┐
                    │   Input IDs  │
                    │ (batch,seq)  │
                    └──────────────┘
```

### 5.2 核心组件定义

#### 5.2.1 配置类

```python
from dataclasses import dataclass

@dataclass
class MiniLLMConfig:
    # 词汇与序列
    vocab_size: int = 8192
    max_seq_len: int = 512

    # 模型维度
    d_model: int = 512          # 隐藏维度
    n_layers: int = 6           # Transformer 层数
    n_heads: int = 8            # 注意力头数
    d_ff: int = 2048            # FFN 中间维度 (d_model × 4)

    # 正则化
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # 位置编码
    use_rotary: bool = True     # RoPE (推荐，性能更优)
    rope_theta: float = 10000.0

    # 激活函数
    activation: str = "gelu"    # GELU / SwiGLU

    # 特殊 Token（不需要在 config 里定义，随 tokenizer 动态读取）
    # pad_token_id / bos_token_id / eos_token_id —— 全由 tokenizer 提供

    # 训练
    use_tied_weights: bool = True  # 权重绑定（LM Head 共享 Embedding 权重）
```

#### 5.2.2 注意力机制（带因果遮罩 + RoPE）

```python
class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力，支持 RoPE 位置编码。
    因果遮罩确保 token i 只能看到 token 0..i（自回归约束）。
    """
    def __init__(self, config: MiniLLMConfig):
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(d_model // n_heads, max_seq_len)

    def forward(self, x, causal_mask=None):
        # Q, K, V 投影 → 分头 → RoPE → 缩放点积注意力 → 合并 → 输出投影
        ...
```

#### 5.2.3 Feed-Forward Network

```python
class FeedForward(nn.Module):
    """
    标准 FFN: Linear → GELU → Linear
    可选升级 SwiGLU（PaLM/LLaMA 风格，需要调整维度）。
    """
    def __init__(self, config):
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()
```

#### 5.2.4 Transformer Block

```python
class TransformerBlock(nn.Module):
    """
    单层 Transformer: Pre-LN → Attention → Residual → Pre-LN → FFN → Residual
    使用 Pre-LayerNorm（现代 GPT/LLaMA 标准，训练更稳定）。
    """
```

#### 5.2.5 完整模型

```python
class MiniLLM(nn.Module):
    """
    完整 Decoder-Only 语言模型:
    Embedding → Transformer Blocks × N → LayerNorm → LM Head
    权重绑定：LM Head 与 Token Embedding 共享权重矩阵
    """
```

### 5.3 参数量估算

```
参数 ≈ vocab_size × d_model                          # Embedding
     + n_layers × (4 × d_model² + 2 × d_model × d_ff) # Transformer
     + d_model × vocab_size                           # LM Head (共享则不计)

以 d_model=512, n_layers=6, d_ff=2048, vocab_size=8192:
  Embedding:    8192 × 512 = 4.2M
  Per Block:    4 × 512² + 2 × 512 × 2048 = 3.1M
  Total Blocks: 6 × 3.1M = 18.9M
  (LM Head 与 Embedding 权重绑定，不加计)
  ─────────────────────────────────────
  Total:        ≈ 23M 参数
  FP32 内存:    ≈ 92MB
```

---

## 6. 阶段四：预训练

### 6.1 训练目标

**自回归语言建模 (Causal Language Modeling)**：给定前文 token 序列，预测下一个 token。

```
Loss = CrossEntropy(predictions, targets)
     = -1/N * Σ log P(token_i | token_0, ..., token_{i-1})
```

### 6.2 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 8 - 16 | CPU 内存限制 |
| Gradient Accumulation | 4 - 8 | 等效增大 batch |
| Learning Rate | 1e-4 → 1e-5 | 带 warmup 的余弦衰减 |
| Warmup Steps | 500 - 1000 | 学习率预热 |
| Max Steps | 10,000 - 50,000 | 总训练步数 |
| Optimizer | AdamW | weight_decay=0.01 |
| LR Schedule | Cosine | 余弦退火至 0 |
| Mixed Precision | 可选 BF16 | 减少 CPU 内存占用 |
| Gradient Clipping | 1.0 | 防止梯度爆炸 |
| Logging Interval | 100 steps | 记录 loss |
| Save Interval | 2000 steps | 保存检查点 |

### 6.3 训练循环伪代码

```python
# 关键组件
model = MiniLLM(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)
scaler = torch.cpu.amp.GradScaler()  # 可选：混合精度

for step in range(max_steps):
    # 1. 获取 batch
    input_ids, labels = next(train_loader)

    # 2. 前向传播
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model(input_ids)          # (B, S, vocab_size)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=pad_token_id     # 忽略 PAD 位置
        )

    # 3. 反向传播
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    # 4. 学习率调整
    scheduler.step()

    # 5. 日志与评估
    if step % log_interval == 0:
        print(f"Step {step}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        # 可选：在验证集上计算困惑度 (Perplexity)
        # ppl = exp(validation_loss)
```

### 6.4 训练监控指标

| 指标 | 计算方式 | 含义 |
|------|----------|------|
| Train Loss | CrossEntropy | 训练损失 |
| Val Loss | CrossEntropy (无 dropout) | 验证损失 |
| **Perplexity** | **exp(Val Loss)** | 模型困惑度，越低越好 |
| 梯度范数 | L2 norm of gradients | 监控梯度健康 |
| Tokens/sec | batch_size × seq_len / time | 训练吞吐量 |

### 6.5 训练策略细节

#### 渐进式序列长度训练（可选）

```
阶段1: seq_len=128, steps=5K  → 快速学习局部模式
阶段2: seq_len=256, steps=5K  → 扩展上下文
阶段3: seq_len=512, steps=10K → 完整上下文训练
```

优势：早期训练更快（注意力计算量 O(n²)），逐步适应长序列。

#### 混合精度训练说明

CPU 上使用 BF16 可以：
- 减少 50% 内存占用
- 可能轻微加速（取决于 CPU 是否支持 AVX-512_BF16）
- 对模型精度影响极小

---

## 7. 阶段五：监督微调（重点）

SFT 是本项目的核心阶段，目标是让一个只会"续写文本"的基础模型获得**指令遵循**和**对话**能力。以下将 SFT 拆分为 6 个子阶段，从数据到评估形成完整闭环。

### 7.0 SFT 子阶段总览

```
┌─────────────┐   ┌─────────────┐   ┌──────────────┐
│ 7.1 数据准备 │ → │ 7.2 模板     │ → │ 7.3 损失掩码  │
│ 下载/清洗    │   │ 设计/转换    │   │ 策略实现      │
└─────────────┘   └─────────────┘   └──────────────┘
                                           │
┌─────────────┐   ┌─────────────┐   ┌──────▼───────┐
│ 7.6 迭代优化 │ ← │ 7.5 评估     │ ← │ 7.4 训练执行  │
│ Bad Case分析 │   │ 对比/人工    │   │ 超参/监控     │
└─────────────┘   └─────────────┘   └──────────────┘
```

### 7.1 子阶段一：SFT 数据准备

#### 7.1.1 数据筛选策略

从 Alpaca 52K 全量数据中筛选高质量子集，而非全量使用：

```python
# 筛选策略
def filter_sft_data(samples, max_samples=10000):
    filtered = []

    for sample in samples:
        # 规则 1：过滤过短或过长的输出（质量信号）
        output_len = len(sample["output"].split())
        if output_len < 10 or output_len > 500:
            continue

        # 规则 2：过滤空 instruction
        if len(sample["instruction"].strip()) == 0:
            continue

        # 规则 3：去重（基于 instruction 的语义相似度或精确匹配）
        # 简单做法：精确去重
        if sample["instruction"] in seen_instructions:
            continue

        # 规则 4：语言一致性检查（可选）
        # 确保 instruction 和 output 语言一致

        filtered.append(sample)

    return filtered[:max_samples]
```

#### 7.1.2 数据集划分

```python
# 划分训练/验证/测试集 (80/10/10)
train_data = filtered[:8000]
val_data   = filtered[8000:9000]
test_data  = filtered[9000:10000]
```

#### 7.1.3 数据多样性检查

确保数据集覆盖多种任务类型：

| 任务类别 | 目标占比 | 示例 |
|---------|---------|------|
| 知识问答 | 25% | "什么是光合作用？" |
| 文本生成 | 20% | "写一首关于春天的诗" |
| 代码生成 | 15% | "写一个 Python 排序函数" |
| 推理分析 | 15% | "分析以下数据的趋势" |
| 文本改写 | 15% | "将以下段落改写得更简洁" |
| 其他 | 10% | 分类、翻译等 |

### 7.2 子阶段二：Prompt 模板设计

#### 7.2.1 模板选型

不同模型使用不同的对话模板，本项目支持多种模板格式，便于对比实验：

**模板 A — Alpaca 风格（简洁）：**

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**模板 B — ChatML 风格（OpenAI 兼容）：**

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{instruction}
{input}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
```

**模板 C — LLaMA 风格（推荐，现代标准）：**

```
<s>[INST] {instruction}
{input} [/INST] {output} </s>
```

**推荐：** 首选模板 A（Alpaca）作为基线，后续可实验模板 C（LLaMA）对比效果。

#### 7.2.2 特殊 Token 与角色标记

```python
# 定义对话角色标记（添加到 tokenizer 的 special_tokens）
SPECIAL_TOKENS = {
    "system_start":  "<|system|>",
    "user_start":    "<|user|>",
    "assistant_start": "<|assistant|>",
    "system_end":    "</s>",     # 复用 EOS
    "user_end":      "</s>",
    "assistant_end": "</s>",
}
```

#### 7.2.3 模板工程核心要点

1. **清晰的角色边界** — 模型需要明确知道当前是 user 还是 assistant 在说话
2. **一致的终止标记** — 生成时遇到终止标记应立即停止
3. **System Prompt** — 可选的系统级指令，定义助手行为
4. **多轮对话支持**（可选）— 将历史轮次拼接为上下文

```python
def format_single_turn(template_type, instruction, input_text, output_text):
    """单轮对话格式化"""
    ...

def format_multi_turn(template_type, messages):
    """多轮对话格式化（进阶）
    messages = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "..."},
    ]
    """
    ...
```

### 7.3 子阶段三：损失掩码策略

这是 SFT 中**最关键的技术细节** —— 如果掩码不正确，模型将学到错误的信号。

#### 7.3.1 核心原则

**只对 assistant 的回复部分计算损失，对 instruction / system prompt / user 输入部分全部屏蔽。**

```
输入序列:
<s> Below is an instruction... ### Instruction: 解释什么是AI ### Response: AI是人工智能的缩写...</s>
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     这些 token 的 label = -100 (忽略)                      这些 token 的 label = 真实 token id
```

#### 7.3.2 掩码实现

```python
def create_sft_labels(input_ids, response_start_pos, pad_token_id=-100):
    """
    构建 SFT 训练标签。

    Args:
        input_ids: 完整的 token 序列 (包含 prompt + response)
        response_start_pos: response 开始的 token 位置索引
        pad_token_id: 忽略位置的填充值（PyTorch 默认为 -100）

    Returns:
        labels: 与 input_ids 等长的标签张量
    """
    labels = input_ids.clone()
    # 将 prompt 部分全部设为 ignore_index
    labels[:, :response_start_pos] = pad_token_id
    return labels

# 更通用的实现：基于角色标记自动定位
def create_sft_labels_auto(input_ids, tokenizer):
    """
    自动根据特殊 token 定位 assistant 回复的起始位置。
    例如对于 ChatML: 找到第一个 <|im_start|>assistant 后的位置。
    """
    assistant_token_id = tokenizer.token_to_id("<|assistant|>")
    labels = input_ids.clone()

    for i in range(input_ids.size(0)):
        # 找到 assistant 标记的位置
        assistant_positions = (input_ids[i] == assistant_token_id).nonzero(as_tuple=True)[0]
        if len(assistant_positions) > 0:
            # response 从 assistant 标记后的下一个 token 开始
            response_start = assistant_positions[0].item() + 1
            labels[i, :response_start] = -100
        else:
            # 未找到 assistant 标记，整行忽略（安全回退）
            labels[i, :] = -100

    return labels
```

#### 7.3.3 损失计算

```python
# PyTorch CrossEntropyLoss 默认 ignore_index=-100
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

# 前向传播
logits = model(input_ids)                          # (B, S, vocab_size)
loss = loss_fn(
    logits.view(-1, vocab_size),                    # (B*S, vocab_size)
    labels.view(-1)                                 # (B*S)
)
# 只有 labels != -100 的位置参与损失计算
```

#### 7.3.4 常见错误与检查

| 错误 | 后果 | 检查方法 |
|------|------|----------|
| 未屏蔽 instruction | 模型学习"背诵"问题 | 观察微调后是否只会复述问题 |
| 屏蔽了 EOS | 模型不会停止生成 | 检查 generation 是否无限循环 |
| 偏移 1 位 | 标签和预测错位 | 打印 input_ids[i] 与 labels[i] 逐位对比 |
| 包含 PAD | 浪费计算 | 确认 -100 = PAD token 的处理方式与预训练一致 |

### 7.4 子阶段四：训练执行

#### 7.4.1 SFT 超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Epochs | 3 - 5 | 小数据集多轮训练 |
| Batch Size | 4 - 8 | 比预训练略小 |
| Learning Rate | 5e-5 - 1e-5 | 比预训练小一个数量级 |
| LR Schedule | Cosine with Warmup | |
| Warmup Ratio | 0.03 - 0.1 | 3%-10% 步数用于 warmup |
| Weight Decay | 0.01 - 0.1 | 适度正则防止过拟合 |
| Max Length | 512 | 与预训练一致 |
| Dropout | 0.05 - 0.1 | 微调阶段略低（减少信息丢失） |

#### 7.4.2 SFT 训练循环

```python
def sft_train(model, train_loader, val_loader, config, logger):
    """
    SFT 训练循环，相比预训练的关键差异：
    1. 使用带掩码的 labels（仅对 response 部分计算 loss）
    2. 更小的学习率
    3. 更早的早停策略（小数据容易过拟合）
    4. 每个 epoch 在验证集上评估
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft_lr,
                                   weight_decay=config.sft_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.sft_epochs * steps_per_epoch)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.sft_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            input_ids = batch['input_ids']
            labels = batch['labels']  # 已预先计算好掩码

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            logger.log_step(loss.item(), scheduler.get_last_lr()[0])

        # Epoch 级验证
        val_loss = evaluate_sft(model, val_loader)
        val_ppl = math.exp(val_loss)
        logger.log_epoch(epoch, train_losses, val_loss, val_ppl)

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, tag="best")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        save_checkpoint(model, optimizer, epoch, val_loss, tag="last")

    return model
```

#### 7.4.3 过拟合监控

小模型 + 小数据集，过拟合是最大风险：

```python
# 每个 epoch 后记录以下指标
metrics = {
    "train_loss": avg_train_loss,
    "val_loss": val_loss,
    "val_ppl": val_ppl,
    "loss_gap": avg_train_loss - val_loss,  # 差距增大=过拟合
    "grad_norm": avg_grad_norm,
}

# 过拟合信号：
# 1. val_loss 不降反升，train_loss 持续下降
# 2. loss_gap > 0.5（经验阈值）
# 3. val_ppl 连续 2 个 epoch 上升
```

### 7.5 子阶段五：评估与对比

#### 7.5.1 自动评估指标

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **Val Loss** | CrossEntropy（仅 response 部分） | 核心指标 |
| **Val PPL** | exp(Val Loss) | 可读性更好的困惑度 |
| **Response Length** | 平均生成 token 数 | 是否过度简洁或冗长 |
| **EOS Rate** | 正确以 EOS 结束的比例 | 模型是否学会了"适可而止" |
| **Repetition** | 生成文本中重复 n-gram 比率 | 检测重复退化 |

#### 7.5.2 对比基线

```python
def compare_models(base_model, sft_model, test_prompts):
    """
    A/B 对比：基础模型 vs SFT 微调模型
    在相同的 prompt 上对比输出，人工或 LLM 评判。
    """
    results = []
    for prompt in test_prompts:
        base_out = generate(base_model, prompt)
        sft_out = generate(sft_model, prompt)
        results.append({
            "prompt": prompt,
            "base_model": base_out,
            "sft_model": sft_out,
        })
    return results
```

#### 7.5.3 人工评估维度（抽取 50-100 条测试）

| 维度 | 评分标准 (1-5) | 权重 |
|------|---------------|------|
| **指令遵循度** | 是否准确理解并执行了指令 | 30% |
| **回答准确性** | 内容是否事实正确 | 25% |
| **语言流畅性** | 是否通顺自然 | 20% |
| **有用性** | 答案是否实用、完整 | 15% |
| **安全性** | 是否有害、偏见内容 | 10% |

#### 7.5.4 评估脚本示例

```python
# 批量评估并生成报告
def run_evaluation(model, test_data, output_dir):
    results = []
    for item in test_data:
        output = generate(model, item['instruction'])
        results.append({
            "instruction": item['instruction'],
            "expected": item['output'],
            "generated": output,
        })

    # 保存评估结果
    with open(f"{output_dir}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 统计指标
    print(f"Avg response length: {avg_len}")
    print(f"EOS rate: {eos_rate:.2%}")
    return results
```

### 7.6 子阶段六：迭代优化

SFT 不是一次性的，而是评估 → 分析 → 优化 → 再训练的循环过程。

#### 7.6.1 Bad Case 分析方法

```python
def analyze_bad_cases(eval_results, threshold=3):
    """
    分析低分样本，分类问题原因。
    """
    bad_cases = []

    for result in eval_results:
        if result['score'] < threshold:
            issue_type = classify_issue(result)
            bad_cases.append({
                "instruction": result['instruction'],
                "generated": result['generated'],
                "expected": result['expected'],
                "issue_type": issue_type,
            })

    # 统计问题分布
    issue_distribution = Counter(c['issue_type'] for c in bad_cases)
    print(f"Bad case distribution: {issue_distribution}")
    return bad_cases
```

#### 7.6.2 常见问题与对策

| 问题类型 | 症状 | 对策 |
|----------|------|------|
| **幻觉** | 编造不存在的事实 | 增加知识密集型数据、降低 temperature |
| **指令误解** | 答非所问 | 增加该任务类型的数据 |
| **回复截断** | max_new_tokens 不够 | 调大 max_tokens 或训练更长回复 |
| **重复生成** | 同一句话反复 | 增大 repetition_penalty、增加多样性数据 |
| **过度拒绝** | 简单问题也回答"无法回答" | 平衡安全数据和有用数据比例 |
| **格式错误** | 要求 JSON 输出但格式不对 | 增加格式约束类训练数据 |

#### 7.6.3 迭代训练策略

```
第 1 轮 SFT：基线训练 (10K 数据, 3 epochs)
    ↓
评估分析：识别 Bad Case 类型
    ↓
第 2 轮 SFT：针对性增加 2K 条该类型数据，从第 1 轮的 best checkpoint 继续训练 (1-2 epochs)
    ↓
评估分析：确认改进效果
    ↓
第 3 轮 SFT（可选）：全量数据混合微调 (1 epoch, 更低 LR)
```

#### 7.6.4 数据飞轮（进阶）

```python
# 将模型自身的生成结果作为训练数据补充
# 仅适用于有高质量评判标准（如代码可执行性、数学可验证性）的场景

# 1. 用 SFT 模型为未标注的 instruction 生成多个候选回复
# 2. 用规则或评分器筛选高质量回复
# 3. 将筛选后的 (instruction, output) 对加入训练集
```

---

## 8. 阶段六：推理与评估

### 8.1 自回归生成算法

```python
def generate(
    model, tokenizer, prompt,
    max_new_tokens=128,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
):
    """
    支持的解码策略:
    - Greedy (temperature=0)
    - Temperature Sampling
    - Top-K Sampling
    - Top-P (Nucleus) Sampling
    - Repetition Penalty（抑制重复）
    """
    input_ids = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)          # (1, seq_len, vocab)
            next_logits = logits[:, -1, :]      # 只取最后位置的 logits

            # 应用 temperature
            next_logits = next_logits / temperature

            # Top-K 过滤
            if top_k > 0:
                topk_values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_values[:, -1:]] = -float('inf')

            # Top-P 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                next_logits.scatter_(1, sorted_indices, sorted_logits.masked_fill(sorted_mask, -float('inf'))

            # 重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_logits[0, token_id] < 0:
                        next_logits[0, token_id] *= repetition_penalty
                    else:
                        next_logits[0, token_id] /= repetition_penalty

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 遇到 EOS 停止
            if next_token.item() == tokenizer.token_to_id("<EOS>"):
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)

    return tokenizer.decode(input_ids[0].tolist())
```

### 8.2 评估指标

| 指标 | 计算方式 | 说明 |
|------|----------|------|
| **Perplexity** | exp(cross_entropy_loss) | 衡量模型对文本的"惊讶度" |
| **生成多样性** | distinct n-grams | 避免重复生成 |
| **生成质量** | 人工评分 | 连贯性、准确性、有用性 |
| **Token 准确率** | Top-1 / Top-5 Accuracy | 预测准确性 |

### 8.3 简易 Web Demo

```python
# 使用 Gradio 快速搭建交互界面
import gradio as gr

def chat(message, history):
    response = generate(model, tokenizer, message)
    return response

gr.ChatInterface(chat).launch()
```

---

## 9. 阶段七：模型冻结与保存

### 9.1 保存内容清单

| 文件 | 格式 | 说明 |
|------|------|------|
| `config.json` | JSON | 模型配置参数（无需外部 vocab 引用） |
| `model.safetensors` | SafeTensors | **推荐格式**：安全、快速、跨平台 |
| `model.pt` | PyTorch Pickle | 备选格式（兼容性更好，但不如 safetensors 安全） |
| `tokenizer.json` | JSON | HuggingFace tokenizer 格式 |
| `training_args.json` | JSON | 训练参数记录 |

### 9.2 保存代码

```python
import safetensors.torch

# =========== 方式一：SafeTensors（推荐） ===========
state_dict = model.state_dict()
state_dict = {k: v.cpu().contiguous() for k, v in state_dict.items()}
safetensors.torch.save_file(state_dict, "output/model.safetensors")

import json
with open("output/config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

# =========== 方式二：完整 PyTorch Checkpoint ===========
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'config': config,
    'step': step,
    'loss': loss.item(),
}, "output/checkpoint_step_10000.pt")

# =========== 方式三：HuggingFace 兼容格式（可选） ===========
model.save_pretrained("output/hf_format")
tokenizer.save("output/hf_format/tokenizer.json")
```

### 9.3 加载代码

```python
# SafeTensors 加载
from safetensors.torch import load_file
state_dict = load_file("output/model.safetensors")
model = MiniLLM(config)
model.load_state_dict(state_dict)
model.eval()

# PyTorch Checkpoint 加载（恢复训练）
checkpoint = torch.load("output/checkpoint_step_10000.pt", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_step = checkpoint['step']
```

### 9.4 模型冻结最佳实践

```python
def freeze_model(model, output_dir):
    """
    冻结模型用于推理部署：
    1. 设置为 eval 模式（禁用 dropout / batch_norm）
    2. 冻结所有参数（不计算梯度）
    3. 保存为标准格式
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 可选：TorchScript 编译（进一步优化 CPU 推理）
    # traced = torch.jit.script(model)
    # traced.save(f"{output_dir}/model_traced.pt")

    safetensors.torch.save_file(
        {k: v.cpu().contiguous() for k, v in model.state_dict().items()},
        f"{output_dir}/model.safetensors"
    )
    print(f"Model frozen and saved to {output_dir}")
```

---

## 10. 项目目录结构

```
LLM/
│
├── configs/                         # 配置文件（按模型规模 + 阶段）
│   ├── model/
│   │   ├── config_10m.yaml          # 10M 参数配置
│   │   ├── config_25m.yaml          # 25M 参数配置（推荐）
│   │   └── config_50m.yaml          # 50M 参数配置
│   ├── pretrain/
│   │   └── default.yaml             # 预训练超参数
│   └── sft/
│       ├── alpaca_template.yaml     # Alpaca 模板配置
│       └── chatml_template.yaml     # ChatML 模板配置
│
├── data/                            # 数据目录（不纳入 Git）
│   ├── raw/                         # 原始下载数据
│   │   ├── wikitext/                # WikiText-103 原始文件
│   │   └── tinystories/             # TinyStories 原始文件
│   ├── processed/                   # 清洗后文本
│   │   └── corpus.txt               # 合并清洗后的训练语料
│   └── tokenized/                   # 分词后的 .bin / .npy 文件
│       ├── train.bin                # 预训练集
│       └── val.bin                  # 预训练验证集
│
├── sft_data/                        # SFT 专用数据（独立于预训练数据）
│   ├── raw/
│   │   └── alpaca_data.json         # 原始 Alpaca 数据
│   ├── processed/
│   │   ├── train.json               # 格式化后的训练集
│   │   ├── val.json                 # 格式化后的验证集
│   │   └── test.json                # 格式化后的测试集
│   └── tokenized/
│       ├── train_sft.bin
│       └── val_sft.bin
│
├── src/                             # 源代码（按阶段组织）
│   ├── __init__.py
│   ├── common/                      # 公共模块（跨阶段复用）
│   │   ├── __init__.py
│   │   ├── config.py                # MiniLLMConfig 定义
│   │   ├── logging_utils.py         # 日志配置与工具
│   │   └── utils.py                 # 通用工具函数
│   │
│   ├── phase1_data/                 # 阶段一：数据工程
│   │   ├── __init__.py
│   │   ├── download.py              # 数据下载
│   │   ├── preprocess.py            # 文本清洗与预处理
│   │   └── dataset.py               # PyTorch Dataset（预训练用）
│   │
│   ├── phase2_tokenizer/            # 阶段二：分词器
│   │   ├── __init__.py
│   │   ├── train_tokenizer.py       # BPE 分词器训练
│   │   └── tokenize_data.py         # 数据批量分词脚本
│   │
│   ├── phase3_model/                # 阶段三：模型架构
│   │   ├── __init__.py
│   │   ├── attention.py             # 因果自注意力 + RoPE
│   │   ├── rotary.py                # RoPE 位置编码
│   │   ├── feedforward.py           # FFN (GELU / SwiGLU)
│   │   ├── transformer_block.py     # Transformer Block
│   │   └── model.py                 # MiniLLM 完整模型
│   │
│   ├── phase4_pretrain/             # 阶段四：预训练
│   │   ├── __init__.py
│   │   ├── trainer.py               # 预训练循环
│   │   ├── metrics.py               # loss / ppl 计算
│   │   └── checkpoint.py            # 检查点保存/加载
│   │
│   ├── phase5_sft/                  # 阶段五：监督微调（重点）
│   │   ├── __init__.py
│   │   ├── data_prepare.py          # 数据筛选、清洗、划分
│   │   ├── templates.py             # Prompt 模板定义（Alpaca / ChatML / LLaMA）
│   │   ├── data_format.py           # 数据格式转换（JSON → tokenized 序列）
│   │   ├── loss_mask.py             # 损失掩码构建
│   │   ├── sft_dataset.py           # SFT PyTorch Dataset
│   │   ├── trainer.py               # SFT 训练循环
│   │   ├── evaluate.py              # 自动评估 + Bad Case 分析
│   │   └── iteration.py             # 迭代优化辅助工具
│   │
│   └── phase6_inference/            # 阶段六：推理部署
│       ├── __init__.py
│       ├── generate.py              # 推理生成（温度/TopK/TopP/重复惩罚）
│       ├── benchmark.py             # 推理速度/内存基准测试
│       └── app.py                   # Gradio 交互界面
│
├── models/                          # 训练产出（模型文件，不纳入 Git）
│   ├── tokenizer/
│   │   └── tokenizer.json           # 训练好的分词器
│   ├── checkpoints/                 # 预训练检查点
│   │   ├── pretrain_step_2000.pt
│   │   ├── pretrain_step_4000.pt
│   │   └── pretrain_best.pt
│   └── sft_checkpoints/             # SFT 检查点
│       ├── sft_epoch_1_best.pt
│       └── sft_epoch_3_best.pt
│
├── output/                          # 最终产出（冻结模型 + 评估报告）
│   ├── final/
│   │   ├── config.json
│   │   ├── model.safetensors        # 最终冻结的模型
│   │   ├── model.pt                 # 备选 PyTorch 格式
│   │   └── tokenizer.json           # 配套分词器
│   └── reports/
│       ├── sft_eval_report.json     # SFT 评估报告
│       ├── bad_cases.json           # Bad Case 分析
│       └── comparison.md            # 基座模型 vs SFT 模型对比
│
├── logs/                            # 日志目录（按阶段分文件）
│   ├── pretrain/
│   │   └── train_20260605_143000.log
│   ├── sft/
│   │   └── train_20260606_100000.log
│   └── inference/
│       └── benchmark_20260606.log
│
├── scripts/                         # 一键运行脚本
│   ├── run_pipeline.sh              # 全流程一键运行
│   ├── run_phase1_data.sh
│   ├── run_phase2_tokenizer.sh
│   ├── run_phase3_test.sh           # 模型单元测试
│   ├── run_phase4_pretrain.sh
│   ├── run_phase5_sft.sh
│   └── run_phase6_demo.sh           # 启动推理 Demo
│
├── notebooks/                       # Jupyter 探索笔记
│   ├── 01_data_explore.ipynb        # 数据分布探索
│   ├── 02_tokenizer_demo.ipynb      # 分词器可视化
│   ├── 03_model_forward.ipynb       # 模型前向传播验证
│   └── 04_sft_analysis.ipynb        # SFT 效果分析
│
├── tests/                           # 单元测试
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_tokenizer.py
│   ├── test_loss_mask.py            # SFT 损失掩码正确性测试
│   └── test_generate.py
│
├── .gitignore                       # 忽略 data/models/output/logs
├── requirements.txt
└── README.md
```

### 10.1 关键目录说明

| 目录 | 用途 | Git | 注释 |
|------|------|-----|------|
| `configs/` | YAML 配置文件（模型参数、训练超参、模板） | 纳入 | 分 model / pretrain / sft 子目录 |
| `src/common/` | 跨阶段复用的公共模块（config、日志、工具函数） | 纳入 | DRY 原则，避免重复 |
| `src/phaseN_*/` | 按阶段划分的源代码模块 | 纳入 | 单一职责，清晰边界 |
| `sft_data/` | SFT 专用数据，独立于预训练数据 | 忽略 | 与预训练数据隔离，便于管理 |
| `models/` | 训练产出的模型文件 | 忽略 | 体积大，通过脚本重新生成 |
| `output/` | 最终冻结模型 + 评估报告 | 忽略 | 报告可选择性纳入 |
| `logs/` | 训练日志（按阶段 + 日期命名） | 忽略 | 运行时生成 |
| `scripts/` | Shell/Python 运行脚本 | 纳入 | 一键执行各阶段 |
| `tests/` | 单元测试 | 纳入 | 保证代码质量 |
| `notebooks/` | Jupyter 探索笔记 | 纳入 | 数据探索与可视化 |

---

## 11. 里程碑与时间估算

| 阶段 | 任务 | 预计耗时 (CPU) | 关键产出 |
|------|------|---------------|----------|
| 1 | 数据下载与清洗 | ~30min | `data/processed/corpus.txt` |
| 2 | 分词器训练 + 数据分词 | ~30min | `models/tokenizer/tokenizer.json`, `data/tokenized/*.bin` |
| 3 | 模型编码与单元测试 | ~2h | `src/phase3_model/` 全部模块可用 |
| 4 | 预训练 (50K steps, 25M) | ~12-48h | 基础语言模型检查点 |
| 5 | 监督微调（含评估） | ~2-5h | 指令模型 + 评估报告 |
| 6 | 推理与 Demo 搭建 | ~30min | Gradio 可交互 Demo |
| 7 | 模型冻结与最终保存 | ~10min | `output/final/model.safetensors` |
| **总计** | | **~2-4 天** | 完整可用的 LLM（含全流程日志） |

---

## 12. 执行指南

### 12.1 环境准备

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 12.2 全流程一键执行

```bash
# ===== 按顺序执行以下命令 =====

# 阶段 1: 数据下载与预处理（~30min）
bash scripts/run_phase1_data.sh
# 或:
python -m src.phase1_data.download
python -m src.phase1_data.preprocess

# 阶段 2: 分词器训练与数据分词（~30min）
bash scripts/run_phase2_tokenizer.sh
# 或:
python -m src.phase2_tokenizer.train_tokenizer
python -m src.phase2_tokenizer.tokenize_data
```

### 12.3 各阶段详细命令

#### 阶段 1+2：数据 + 分词器

```bash
# 下载公开数据集（WikiText-103 + TinyStories）
python -m src.phase1_data.download

# 文本清洗与语料构建
python -m src.phase1_data.preprocess

# 训练 BPE 分词器（词汇表 8192）
python -m src.phase2_tokenizer.train_tokenizer

# 对语料进行批量分词并保存为 .bin 文件
python -m src.phase2_tokenizer.tokenize_data
```

**产出文件：**
- `data/processed/corpus.txt` — 清洗后的训练语料
- `models/tokenizer/tokenizer.json` — BPE 分词器
- `data/tokenized/train.bin` — 预训练集（uint16 numpy memmap）
- `data/tokenized/val.bin` — 验证集

#### 阶段 3：模型架构验证

```bash
# 运行模型单元测试（验证所有组件正确性）
python tests/test_model.py
python tests/test_loss_mask.py
python tests/test_generate.py
```

**预期输出：** 21 个测试全部 `[PASS]`

#### 阶段 4：预训练

```bash
# 使用默认配置启动预训练（25M 参数模型）
python -m src.phase4_pretrain.trainer \
    --model_config configs/model/config_25m.yaml \
    --pretrain_config configs/pretrain/default.yaml

# 可选参数:
#   --resume             从 latest checkpoint 恢复训练
#   --seed 42            设置随机种子
#   --device cpu         指定设备（默认自动检测）
```

**训练监控指标（TensorBoard）：**
```bash
tensorboard --logdir logs/pretrain
```

**产出文件：**
- `models/checkpoints/pretrain_latest.pt` — 最新检查点
- `models/checkpoints/pretrain_best.pt` — 最佳检查点（验证损失最低）
- `models/checkpoints/pretrain_step_*.pt` — 周期检查点
- `logs/pretrain/train_*.log` — 训练日志

#### 阶段 5：监督微调（SFT）

```bash
# 5.1 准备 SFT 数据（下载 Alpaca → 筛选 → 划分 → 分词）
python -m src.phase5_sft.data_prepare

# 5.2 启动 SFT 训练（从预训练 checkpoint 加载）
python -m src.phase5_sft.trainer \
    --model_config configs/model/config_25m.yaml \
    --sft_config configs/sft/default.yaml \
    --pretrained_checkpoint models/checkpoints/pretrain_best.pt

# 5.3 评估 SFT 模型
python -m src.phase5_sft.evaluate \
    --model_config configs/model/config_25m.yaml \
    --sft_checkpoint models/sft_checkpoints/sft_best.pt \
    --test_data sft_data/processed/test.json
```

**产出文件：**
- `sft_data/processed/train.json` — SFT 训练集
- `sft_data/tokenized/train_sft.bin` — 分词后训练数据
- `models/sft_checkpoints/sft_best.pt` — 最佳 SFT 模型
- `output/reports/sft_eval_report.json` — 评估报告
- `logs/sft/train_*.log` — SFT 训练日志

#### 阶段 6：推理与部署

```bash
# 6.1 运行推理基准测试
python -m src.phase6_inference.benchmark \
    --model_config configs/model/config_25m.yaml \
    --model_path models/sft_checkpoints/sft_best.pt

# 6.2 启动 Gradio 交互界面
python -m src.phase6_inference.app \
    --model_config configs/model/config_25m.yaml \
    --model_path models/sft_checkpoints/sft_best.pt
# 访问 http://localhost:7860
```

### 12.4 模型冻结与最终保存

```bash
# 将训练好的模型冻结为 SafeTensors 格式（仅推理用）
python -c "
import torch
from safetensors.torch import save_file
from src.common.config import MiniLLMConfig
from src.phase3_model.model import MiniLLM

# 加载模型
config = MiniLLMConfig.from_yaml('configs/model/config_25m.yaml')
model = MiniLLM(config)
checkpoint = torch.load('models/sft_checkpoints/sft_best.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 冻结参数
for p in model.parameters():
    p.requires_grad = False

# 保存
state_dict = {k: v.cpu().contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, 'output/final/model.safetensors')

import json, shutil
json.dump(config.to_dict(), open('output/final/config.json', 'w'), indent=2)
shutil.copy('models/tokenizer/tokenizer.json', 'output/final/tokenizer.json')
print('Model frozen and saved to output/final/')
"
```

### 12.5 换用不同模型规模

```bash
# 10M 小模型（更快，适合快速实验）
python -m src.phase4_pretrain.trainer \
    --model_config configs/model/config_10m.yaml

# 50M 大模型（更慢，效果更好）
python -m src.phase4_pretrain.trainer \
    --model_config configs/model/config_50m.yaml
```

### 12.6 项目当前状态验证

```bash
# 验证项目结构完整性
python -c "
import sys
modules = [
    'src.common.config',
    'src.common.logging_utils',
    'src.common.utils',
    'src.phase1_data.download',
    'src.phase1_data.preprocess',
    'src.phase1_data.dataset',
    'src.phase2_tokenizer.train_tokenizer',
    'src.phase2_tokenizer.tokenize_data',
    'src.phase3_model.attention',
    'src.phase3_model.rotary',
    'src.phase3_model.feedforward',
    'src.phase3_model.transformer_block',
    'src.phase3_model.model',
    'src.phase4_pretrain.metrics',
    'src.phase4_pretrain.checkpoint',
    'src.phase4_pretrain.trainer',
    'src.phase5_sft.templates',
    'src.phase5_sft.data_prepare',
    'src.phase5_sft.data_format',
    'src.phase5_sft.loss_mask',
    'src.phase5_sft.sft_dataset',
    'src.phase5_sft.trainer',
    'src.phase5_sft.evaluate',
    'src.phase5_sft.iteration',
    'src.phase6_inference.generate',
    'src.phase6_inference.benchmark',
    'src.phase6_inference.app',
]
failed = []
for m in modules:
    try:
        __import__(m)
        print(f'  [OK] {m}')
    except Exception as e:
        print(f'  [FAIL] {m}: {e}')
        failed.append(m)

if failed:
    print(f'\n{len(failed)} module(s) failed to import')
else:
    print(f'\nAll {len(modules)} modules imported successfully')
"
```

---

## 附录

```
torch>=2.0.0
tokenizers>=0.15.0
datasets>=2.14.0
safetensors>=0.4.0
numpy
tqdm
tensorboard
gradio
pyyaml
```

## 附录 B：关键设计决策记录

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 架构 | Decoder-Only (GPT) | 现代 LLM 主流，简单高效 |
| 位置编码 | RoPE | 外推性好，LLaMA/Qwen 标准 |
| 激活函数 | GELU | 简单稳定（小模型不需要 SwiGLU 复杂度） |
| 归一化位置 | Pre-LayerNorm | 训练更稳定，现代标准 |
| 权重绑定 | 是 | 减少 Embedding 参数量 |
| 分词算法 | BPE (Byte-Level) | 无 OOV，GPT 标准 |
| 模型格式 | SafeTensors | 安全、快速、跨平台 |
| 训练框架 | 纯 PyTorch 循环 | 理解每个细节，不依赖黑盒 |
| SFT 模板 | Alpaca (基线) + 可切换 | 简单基线，后期可对比其他模板 |
| 推理界面 | Gradio | 比 Flask 更简单，专为 ML Demo 设计 |

## 附录 C：SFT 常见问题排查清单

| 现象 | 可能原因 | 排查步骤 |
|------|----------|----------|
| SFT 后模型只会输出 EOS | 损失掩码错误，全部 labels 都设为 -100 | 检查 `create_sft_labels` 返回的 labels |
| 模型输出与 instruction 无关 | 损失掩码未正确屏蔽 instruction 部分 | 打印一组 input_ids 和 labels 逐位对比 |
| Val loss 极低但生成质量差 | 过拟合到训练集模板格式 | 检查 `loss_gap`，增大 dropout 或减少 epochs |
| 生成结果总是重复 | repetition_penalty 太低或训练数据多样性不足 | 增大 repetition_penalty，检查数据多样性 |
| EOS 后还有多余文本 | EOS token 未被正确屏蔽 | 确认 EOS 位置在 labels 中不是 -100 |
| 中文和英文混合输出 | 预训练数据语言混杂 | 确保 SFT 数据语言与预训练数据主导语言一致 |
