# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

MiniLLM — 从零训练 GPT-like 解码器专用 Transformer 的教学项目。纯 CPU 可运行（8GB+ RAM），参数量 10M-50M。技术栈：PyTorch + HuggingFace Tokenizers + Gradio + TensorBoard。

## 常用命令

```bash
# 环境初始化
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 运行测试（所有测试直接 python 执行，无 pytest）
python tests/test_model.py          # 9 个模型组件测试
python tests/test_generate.py       # 6 个生成策略测试
python tests/test_loss_mask.py      # 5 个 SFT 损失掩码测试

# 手动运行各阶段（shell 脚本缺少 CLI 参数，需手动传参）
python -m src.phase1_data.download
python -m src.phase1_data.preprocess
python -m src.phase2_tokenizer.train_tokenizer
python -m src.phase2_tokenizer.tokenize_data
python -m src.phase4_pretrain.trainer --model_config configs/model/config_25m.yaml --pretrain_config configs/pretrain/default.yaml
python -m src.phase5_sft.data_prepare [--parquet_path data/sft_data/raw/alpaca.parquet]
python -m src.phase5_sft.trainer --model_config configs/model/config_25m.yaml --sft_config configs/sft/default.yaml --pretrained_checkpoint models/checkpoints/pretrain_best.pt
python -m src.phase5_sft.evaluate --checkpoint models/checkpoints/sft/best.pt
python -m src.phase6_inference.app --model_config configs/model/config_25m.yaml --model_path models/final/model.safetensors --tokenizer_path models/tokenizer/tokenizer.json
python -m src.phase6_inference.benchmark --model_config configs/model/config_25m.yaml --model_path models/final/model.safetensors
```

## 核心架构

### 六阶段流水线

```
phase1_data → phase2_tokenizer → phase3_model → phase4_pretrain → phase5_sft → phase6_inference
```

各阶段在 `src/` 下是独立 Python 包，通过 `src/common/` 共享配置和工具。阶段间依赖：
- `phase4_pretrain` 依赖 `phase3_model` + `phase1_data`
- `phase5_sft` 依赖 `phase3_model`
- `phase6_inference` 依赖 `phase3_model`

### 模型架构（Pre-LN Transformer）

```
Input (B,S) → Embedding → [LayerNorm → Attention(+RoPE) → Residual → LayerNorm → FFN → Residual] × N → Final LayerNorm → LM Head → (B,S,vocab_size)
```

关键设计决策：
- **Pre-LN** 架构（`src/phase3_model/transformer_block.py`）
- **RoPE** 位置编码（`src/phase3_model/rotary.py`），GPT-NeoX 风格配对旋转
- **权重绑定**（`TiedLinear`）：LM Head 与 Token Embedding 共享权重。`TiedLinear` 持有对 embedding weight 的引用（非 `nn.Parameter`），通过 `F.linear(x, self.weight)` 计算，避免重复存储
- **FFN 激活**可选 GELU / SiLU / SwiGLU（`feedforward.py`）
- 注意力使用**组合 QKV 投影**：单次 `nn.Linear(d_model, 3*d_model)` 投影后 `chunk(3)` 拆分
- **GPT-2 风格初始化**：`normal(std=0.02)` for Linear/Embedding，bias 置零，LayerNorm weight=1/bias=0

### 配置系统

所有配置通过 `src/common/config.py` 中 4 个 `@dataclass` 管理，均支持 `from_yaml(path)` 工厂方法和 `to_dict()`：
- `DataConfig` — HuggingFace token
- `MiniLLMConfig` — 模型超参数，`__post_init__` 强制 `d_model % n_heads == 0` 并自动计算 `head_dim = d_model // n_heads`
- `PretrainConfig` — 预训练超参数（warmup + cosine LR schedule，含 `betas`/`eps`/`gradient_accumulation_steps`）
- `SFTConfig` — SFT 超参数（warmup ratio + 早停），`template_type` 字段控制模板选择（alpaca/chatml/llama）

配置文件位于 `configs/{model,pretrain,sft}/`，`config_25m.yaml` 为推荐默认。所有 `from_yaml` 方法会静默过滤 YAML 中的未知字段。

### 数据处理

所有语料使用 `numpy.memmap` 存储为 `.bin` 文件，避免一次性加载到内存。`src/phase1_data/dataset.py` 定义 `PretrainDataset` + `create_dataloader`。分词器为字节级 BPE，词汇表 8192，使用 HuggingFace `tokenizers` 库训练。

### SFT 数据与损失掩码

SFT 数据统一放在 `data/sft_data/` 下（被 `data/` 的 `.gitignore` 覆盖，无需单独配置）：
- `data/sft_data/raw/` — 原始数据（`.parquet` 或 `.json`）
- `data/sft_data/processed/` — 过滤划分后的 train/val/test JSON
- `data/sft_data/tokenized/` — 分词后二进制文件

`data_prepare.py` 支持两种数据来源：
1. 从 HuggingFace 在线下载（默认行为）
2. 通过 `--parquet_path` 从本地 parquet 加载

**SFT 二进制数据格式**：不同于预训练的单 `.bin` 文件，SFT 使用双文件存储：
- `data.bin` — `(N, max_seq_len)` uint16 memmap，存储完整 token 序列
- `prompt_lens.npy` — `(N,)` int32，每个样本的 prompt 长度

`SFTDataset` 在 `__getitem__` 时动态创建 labels：前 `prompt_len` 个 token 设为 -100，padding 位置也设为 -100。

`src/phase5_sft/loss_mask.py` — SFT 的核心机制：只对 assistant 回复部分计算损失（prompt 部分标签设为 -100）。模板系统（`templates.py`）支持 Alpaca / ChatML / LLaMA 三种格式，通过 `PromptTemplate.format_full()` 区分 prompt 和 response 区域。

### 推理生成

项目中存在两套生成实现，功能互补：
- **`model.py:MiniLLM.generate()`** — 模型内置方法，支持 temperature / top-k / top-p 采样，greedy 解码（temperature≤0），EOS 自动停止
- **`phase6_inference/generate.py:TextGenerator`** — 独立生成器类，额外支持 repetition_penalty、stop_tokens 字符串匹配、批量生成、终端交互式聊天（`chat_loop`）

## 注意事项

- **Shell 脚本是占位符**：`scripts/*.sh` 缺少所需的 CLI 参数，不能直接运行。请使用上述手动 `python -m` 命令。
- **无 Makefile / pytest**：项目仅使用直接 `python` 调用，测试文件需从项目根目录运行。
- **部分功能未实现**：`create_pretrain_datasets` 函数在 `pretrain/trainer.py` 中被引用但 `dataset.py` 中未定义；`scripts/run_phase3_test.sh` 和 `scripts/run_pipeline.sh` 在 DESIGN.md 中提及但不存在。
- 项目是教学导向的，所有训练循环均手写（无 HuggingFace Trainer 封装），以透明展示每个步骤。
