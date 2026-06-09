# 服务器 SFT 微调文件传输指南

## 背景

仓库 `.gitignore` 忽略了 `data/`、`models/`、`logs/` 目录，因此 `git clone` 后这些文件不会出现在服务器上，需要手动传输。

---

## 一、需要上传到服务器的文件

### 必需上传（SFT 训练的最小依赖）

| 文件 | 大小 | 用途 |
|------|------|------|
| `models/tokenizer/tokenizer.json` | 553 KB | 分词器 |
| `models/checkpoints/pretrain_best.pt` | 95 MB | 预训练权重 |
| `data/sft_data/raw/alpaca.parquet` | 24 MB | 原始 SFT 数据 |

### 可选上传（跳过数据预处理步骤）

如果不想在服务器上重新跑 `data_prepare.py`，可以直接上传已处理好的 JSON：

| 文件 | 大小 |
|------|------|
| `data/sft_data/processed/train.json` | 2 MB |
| `data/sft_data/processed/val.json` | 259 KB |
| `data/sft_data/processed/test.json` | 252 KB |

### 不需要上传的文件

- `models/checkpoints/pretrain_step_*.pt` — 中间预训练检查点（每个 ~280 MB），SFT 不需要
- `models/checkpoints/pretrain_latest.pt` — 最新检查点（含 optimizer/scheduler），太大且 SFT 不需要
- `models/sft_checkpoints/` — 旧 SFT 检查点

---

## 二、上传命令

### 方式一：rsync（推荐）

```bash
# 在本地项目根目录执行

# 1. 创建远程目录
ssh root@ecs-efbe "mkdir -p ~/work/LLM/models/tokenizer ~/work/LLM/models/checkpoints ~/work/LLM/data/sft_data"

# 2. 上传 tokenizer 和预训练权重
rsync -avz --progress \
  models/tokenizer/tokenizer.json \
  models/checkpoints/pretrain_best.pt \
  root@ecs-efbe:~/work/LLM/

# 3. 上传 SFT 数据（整个 data/sft_data/ 目录）
rsync -avz --progress \
  data/sft_data/ \
  root@ecs-efbe:~/work/LLM/data/sft_data/
```

### 方式二：scp 逐文件

```bash
# tokenizer
scp models/tokenizer/tokenizer.json root@ecs-efbe:~/work/LLM/models/tokenizer/

# 预训练权重
scp models/checkpoints/pretrain_best.pt root@ecs-efbe:~/work/LLM/models/checkpoints/

# SFT 数据
scp -r data/sft_data/ root@ecs-efbe:~/work/LLM/data/
```

---

## 三、服务器上的目录结构

上传完成后，服务器目录应如下：

```
~/work/LLM/
├── models/
│   ├── tokenizer/tokenizer.json        # 553 KB
│   └── checkpoints/pretrain_best.pt   # 95 MB
├── data/sft_data/
│   ├── raw/alpaca.parquet              # 24 MB（原始数据）
│   └── processed/                      # 如果上传了预处理数据
│       ├── train.json
│       ├── val.json
│       └── test.json
├── configs/                            # git 已有
├── src/                                # git 已有
├── tests/                              # git 已有
├── scripts/                            # git 已有
├── CLAUDE.md                           # git 已有
├── README.md                           # git 已有
└── requirements.txt                    # git 已有
```

---

## 四、训练完成后需要下载回来的文件

| 文件/目录 | 说明 | 重要性 |
|-----------|------|--------|
| `models/checkpoints/sft/sft_best.pt` | 最佳模型（早停选出的） | **必须** |
| `models/checkpoints/sft/sft_final.pt` | 最终模型（仅模型权重，文件更小） | **必须** |
| `models/checkpoints/sft/sft_epoch_*.pt` | 各 epoch 检查点（含 optimizer/scheduler，可用于恢复训练） | 可选 |
| `logs/sft/sft_train.log` | 训练日志（文本，方便查看训练过程） | 推荐 |
| `logs/sft/sft_train_metrics.jsonl` | 训练指标 JSONL（方便画 loss 曲线） | 推荐 |

### 下载命令

```bash
# 在本地执行

# 1. 下载 SFT 检查点
rsync -avz --progress \
  root@ecs-efbe:~/work/LLM/models/checkpoints/sft/ \
  models/checkpoints/sft/

# 2. 下载训练日志
rsync -avz --progress \
  root@ecs-efbe:~/work/LLM/logs/sft/ \
  logs/sft/
```

---

## 五、服务器端执行 SFT 训练

```bash
cd ~/work/LLM

# 1. 安装依赖
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. 如果只上传了原始 parquet，需要先做数据预处理
python -m src.phase5_sft.data_prepare --parquet_path data/sft_data/raw/alpaca.parquet

# 3. SFT 训练
python -m src.phase5_sft.trainer \
    --model_config configs/model/config_25m.yaml \
    --sft_config configs/sft/default.yaml \
    --pretrained_checkpoint models/checkpoints/pretrain_best.pt
```

训练完成后，检查点保存在 `models/checkpoints/sft/`，日志保存在 `logs/sft/`。

---

## 六、文件大小参考

| 传输方向 | 内容 | 大小 |
|----------|------|------|
| 上传 | tokenizer + 预训练权重 + 数据 | ~120 MB |
| 下载 | SFT 检查点（3 epoch）+ 日志 | ~500 MB（取决于 epoch 数） |

- `sft_best.pt` / `sft_epoch_*.pt` 含 optimizer + scheduler 状态，每个约 95 MB（与模型大小相同）
- `sft_final.pt` 仅含模型权重，约 95 MB
- 日志文件很小（几 MB 以内）
