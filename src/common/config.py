from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class DataConfig:
    """Configuration for data downloading."""

    # Hugging Face
    hf_token: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class MiniLLMConfig:
    """Configuration for MiniLLM model."""

    # Vocabulary & Sequence
    vocab_size: int = 8192
    max_seq_len: int = 512
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Model dimensions
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048

    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5

    # Position encoding
    use_rotary: bool = True
    rope_theta: float = 10000.0

    # Activation
    activation: Literal["gelu", "silu", "swiglu"] = "gelu"

    # Weight tying
    use_tied_weights: bool = True

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str) -> "MiniLLMConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class PretrainConfig:
    """Configuration for pretraining."""

    # Data
    train_data_path: str = "data/tokenized/train.bin"
    val_data_path: str = "data/tokenized/val.bin"

    # Training
    max_steps: int = 50000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    min_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    use_amp: bool = False  # Mixed precision (BF16 on CPU)

    # Logging & Saving
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 2000
    checkpoint_dir: str = "models/checkpoints"
    log_dir: str = "logs/pretrain"

    # Optimization
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8

    @classmethod
    def from_yaml(cls, path: str) -> "PretrainConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning."""

    # Data
    train_data_path: str = "data/sft_data/tokenized/train_sft.bin"
    val_data_path: str = "data/sft_data/tokenized/val_sft.bin"
    template_type: Literal["alpaca", "chatml", "llama"] = "alpaca"

    # Training
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    max_seq_len: int = 512

    # Overfitting prevention
    dropout: float = 0.05
    early_stop_patience: int = 2

    # Logging & Saving
    log_interval: int = 10
    eval_interval: int = 200
    checkpoint_dir: str = "models/sft_checkpoints"
    log_dir: str = "logs/sft"

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        valid_fields = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}
