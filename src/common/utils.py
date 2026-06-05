"""
General-purpose utility functions for the MiniLLM project.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Additional determinism settings (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def get_device(verbose: bool = True) -> torch.device:
    """Return the best available torch device.

    Currently CPU-only by design.  Displays an informative message.

    Args:
        verbose: If ``True``, print device information.

    Returns:
        ``torch.device("cpu")``.
    """
    device = torch.device("cpu")
    if verbose:
        print(f"[Device] Using CPU. PyTorch version: {torch.__version__}")
    return device


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory(
    model: nn.Module, batch_size: int, seq_len: int
) -> Dict[str, float]:
    """Estimate approximate memory usage for a forward + backward pass in MB.

    This is a rough heuristic:

    - **model_params_mb**: memory occupied by model parameters (fp32).
    - **optimizer_mb**: 2x model params (Adam stores m and v).
    - **activation_mb**: rough estimate proportional to
      ``batch_size * seq_len * d_model * n_layers``.
    - **total_mb**: sum of the above.

    Args:
        model: The PyTorch module.
        batch_size: Micro-batch size.
        seq_len: Sequence length.

    Returns:
        Dictionary with keys ``model_params_mb``, ``optimizer_mb``,
        ``activation_mb``, ``total_mb``.
    """
    num_params = count_parameters(model)
    bytes_per_param = 4  # fp32

    model_mb = (num_params * bytes_per_param) / (1024 ** 2)
    optimizer_mb = 2 * model_mb  # Adam stores two momentum buffers

    # Heuristic activation memory: roughly proportional to the largest tensor
    # shape in the transformer stack.
    d_model = getattr(getattr(model, "config", None), "d_model", 512)
    n_layers = getattr(getattr(model, "config", None), "n_layers", 6)
    activation_mb = (
        batch_size * seq_len * d_model * n_layers * bytes_per_param
    ) / (1024 ** 2)

    total_mb = model_mb + optimizer_mb + activation_mb

    return {
        "model_params_mb": round(model_mb, 2),
        "optimizer_mb": round(optimizer_mb, 2),
        "activation_mb": round(activation_mb, 2),
        "total_mb": round(total_mb, 2),
    }


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def format_time(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string.

    Examples:
        ``format_time(62) -> "1m 2s"``,
        ``format_time(3661) -> "1h 1m 1s"``.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string.
    """
    seconds = max(0, int(seconds))
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)

    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def load_tokenizer(path: str) -> Any:
    """Load a HuggingFace tokenizer from a JSON file and set special token IDs.

    The loaded tokenizer will have ``pad_token_id``, ``bos_token_id``, and
    ``eos_token_id`` injected as attributes if they are available.

    Args:
        path: Path to a ``tokenizer.json`` file.

    Returns:
        A HuggingFace :class:`tokenizers.Tokenizer` instance.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(path)

    # Ensure special token IDs are accessible as attributes on the tokenizer
    # object for convenience.
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}

    if not hasattr(tokenizer, "pad_token_id"):
        tokenizer.pad_token_id = vocab.get("<pad>", 0)  # type: ignore[attr-defined]
    if not hasattr(tokenizer, "bos_token_id"):
        tokenizer.bos_token_id = vocab.get("<bos>", 1)  # type: ignore[attr-defined]
    if not hasattr(tokenizer, "eos_token_id"):
        tokenizer.eos_token_id = vocab.get("<eos>", 2)  # type: ignore[attr-defined]

    return tokenizer


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary as a JSON file.

    Creates parent directories if they do not exist.

    Args:
        data: Dictionary to serialize.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict[str, Any]:
    """Load a dictionary from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Deserialized dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return a Path object.

    Args:
        path: Directory path to create.

    Returns:
        Path object for the created directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def setup_logging(
    name: str = "mini_llm",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up structured logging with console and file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_path / f"{name}_{timestamp}.log", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def format_number(n: int) -> str:
    """Format a large integer with human-readable suffixes.

    Args:
        n: Integer to format.

    Returns:
        Formatted string (e.g. '1.23M', '4.56K').
    """
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)
