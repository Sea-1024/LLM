# MiniLLM

从零训练 GPT-like 解码器专用 Transformer 的教学项目。纯 CPU 可运行（8GB+ RAM），参数量 10M-50M，完整覆盖数据准备→分词→预训练→SFT→推理的全流程。

## 快速开始

```bash
# 环境准备
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 运行测试验证环境
python tests/test_model.py
python tests/test_generate.py
python tests/test_loss_mask.py
```

## 项目结构

```
LLM/
├── configs/                     # 配置文件
│   ├── model/                   #   模型超参数（10M / 25M / 50M）
│   ├── pretrain/                #   预训练超参数
│   └── sft/                     #   SFT 超参数
├── src/
│   ├── common/                  # 公共工具（配置、日志、工具函数）
│   ├── phase1_data/             # 阶段一：数据下载与预处理
│   ├── phase2_tokenizer/        # 阶段二：分词器训练与数据分词
│   ├── phase3_model/            # 阶段三：模型架构（Attention/RoPE/FFN/Transformer）
│   ├── phase4_pretrain/         # 阶段四：预训练（训练循环/检查点/指标）
│   ├── phase5_sft/              # 阶段五：SFT 微调（模板/数据格式/训练/评估/迭代工具）
│   └── phase6_inference/        # 阶段六：推理（文本生成/聊天/Gradio 应用/基准测试）
├── tests/                       # 测试（纯 Python，无 pytest）
├── scripts/                     # Shell 脚本（占位符，不可直接使用）
└── models/                      # 模型产物（检查点、最终模型、分词器、日志）
```

## 完整训练流程

### 第一阶段：数据准备

```bash
# 下载 WikiText-103 数据集
python -m src.phase1_data.download

# 预处理为纯文本格式
python -m src.phase1_data.preprocess
```

### 第二阶段：分词器训练

```bash
# 训练 BPE 分词器（词汇量 8192）
python -m src.phase2_tokenizer.train_tokenizer

# 将文本语料分词为二进制格式（memmap）
python -m src.phase2_tokenizer.tokenize_data
```

### 第三阶段：模型定义

无需 CLI——模型代码作为库通过 `from src.phase3_model.model import MiniLLM` 使用。

### 第四阶段：预训练

```bash
python -m src.phase4_pretrain.trainer \
    --model_config configs/model/config_25m.yaml \
    --pretrain_config configs/pretrain/default.yaml
```

关键特性：梯度累积、warmup + cosine 学习率调度、定期验证与检查点保存、TensorBoard 日志。

### 第五阶段：SFT 微调

```bash
# 数据准备（从 HuggingFace 下载 Alpaca 数据集）
python -m src.phase5_sft.data_prepare

# SFT 训练
python -m src.phase5_sft.trainer \
    --model_config configs/model/config_25m.yaml \
    --sft_config configs/sft/default.yaml \
    --pretrained_checkpoint models/checkpoints/pretrain_best.pt

# 评估
python -m src.phase5_sft.evaluate \
    --model_config configs/model/config_25m.yaml \
    --checkpoint models/sft_checkpoints/sft_best.pt \
    --test_data data/sft_data/processed/test.json

# 与基础模型对比
python -m src.phase5_sft.evaluate \
    --model_config configs/model/config_25m.yaml \
    --checkpoint models/sft_checkpoints/sft_best.pt \
    --test_data data/sft_data/processed/test.json \
    --compare_base_checkpoint models/checkpoints/pretrain_best.pt
```

SFT 支持三种对话模板：`alpaca`（默认）、`chatml`、`llama`，通过 `configs/sft/*.yaml` 中的 `template_type` 切换。

### 第六阶段：推理

```bash
# 交互式聊天（Gradio 网页界面）
python -m src.phase6_inference.app \
    --model_config configs/model/config_25m.yaml \
    --model_path models/final/model.safetensors \
    --tokenizer_path models/tokenizer/tokenizer.json

# 基准测试（吞吐量、延迟）
python -m src.phase6_inference.benchmark \
    --model_config configs/model/config_25m.yaml \
    --model_path models/final/model.safetensors
```

## 模型架构

```
Input → Embedding → [LayerNorm → CausalAttention(+RoPE) → Residual
                  → LayerNorm → FFN(GELU/SiLU/SwiGLU) → Residual] × N
                  → Final LayerNorm → TiedLMHead → Output
```

| 特性 | 实现 |
|------|------|
| 架构 | Pre-LN Decoder-only Transformer |
| 位置编码 | RoPE（GPT-NeoX 风格） |
| 注意力 | 组合 QKV 投影 + Causal Mask |
| 激活函数 | GELU / SiLU / SwiGLU 可选 |
| 权重绑定 | LM Head 与 Token Embedding 共享 |
| 初始化 | GPT-2 风格（normal std=0.02） |

## 模型配置

| 配置 | 参数量 | d_model | n_layers | n_heads | d_ff |
|------|--------|---------|----------|---------|------|
| `config_10m.yaml` | ~10M | 384 | 4 | 6 | 1536 |
| `config_25m.yaml` | ~25M | 512 | 6 | 8 | 2048 |
| `config_50m.yaml` | ~50M | 768 | 8 | 12 | 3072 |

## 配置系统

所有配置通过 `@dataclass` 管理，支持 YAML 加载：

```python
from src.common.config import MiniLLMConfig, PretrainConfig, SFTConfig

model_cfg = MiniLLMConfig.from_yaml("configs/model/config_25m.yaml")
pretrain_cfg = PretrainConfig.from_yaml("configs/pretrain/default.yaml")
```

YAML 中的未知字段会被静默忽略，避免配置污染。

## 生成策略

项目提供两套生成实现：

- **`MiniLLM.generate()`** — 模型内置方法，支持 temperature / top-k / top-p 采样、greedy 解码、EOS 停止
- **`TextGenerator`** — 独立生成器，额外支持 repetition_penalty、stop_tokens、批量生成、交互式聊天

## 技术栈

| 组件 | 用途 |
|------|------|
| PyTorch >= 2.0 | 核心框架 |
| HuggingFace Tokenizers | BPE 分词器训练与推理 |
| HuggingFace Datasets | 数据集下载 |
| Gradio | 交互式网页推理界面 |
| TensorBoard | 训练监控 |
| safetensors | 模型序列化 |
| NumPy memmap | 大规模数据零拷贝加载 |

## 设计要点

- **教学透明**：所有训练循环手写，无 HuggingFace Trainer 封装
- **纯 CPU 可行**：25M 模型在 8GB RAM CPU 上可完成全流程训练
- **损失掩码**：SFT 仅对 assistant 回复计算损失，prompt 区域自动屏蔽
- **检查点完整**：支持中断恢复、最佳模型追踪、预训练权重热加载到 SFT
- **早停机制**：基于验证损失，patience 可配置（默认 2-3 轮）

## 测试

```bash
python tests/test_model.py       # 模型组件：配置、RoPE、注意力、FFN、TransformerBlock、前向传播、生成、权重绑定、参数量
python tests/test_generate.py    # 生成策略：基础采样、greedy、top-k、top-p、截断、EOS 停止
python tests/test_loss_mask.py   # SFT 损失掩码：标签创建、批处理、边界情况、CrossEntropy 忽略
```

测试无需额外测试框架，直接 `python` 运行即可。

## 注意事项

- `scripts/*.sh` 是占位符，缺少所需 CLI 参数，请使用 `python -m` 命令
- 项目无 Makefile、无 pytest——所有操作通过直接 `python` 调用完成
- 更多实现细节见 [CLAUDE.md](CLAUDE.md)
