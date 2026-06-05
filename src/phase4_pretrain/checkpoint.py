"""
Checkpoint save/load utilities for pretraining.

Supports saving and resuming full training state (model, optimizer, scheduler,
scaler, step, loss, config) and finding the latest or best checkpoint.
"""

import os
from typing import Any, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    loss: float,
    config: Any,
    checkpoint_dir: str,
    tag: str = "step",
    scaler: Optional[Any] = None,
) -> str:
    """
    Save a complete training checkpoint with all state needed to resume training.

    Produces two files:
      - {checkpoint_dir}/pretrain_{tag}_{step}.pt   (timestamped)
      - {checkpoint_dir}/pretrain_latest.pt          (overwritten each call)

    Args:
        model: The MiniLLM model.
        optimizer: AdamW optimizer instance.
        scheduler: LR scheduler instance.
        step: Current training step (global).
        loss: Current loss value.
        config: Model config object (must have to_dict or __dict__).
        checkpoint_dir: Directory to save checkpoints into.
        tag: Label for the checkpoint file (e.g. "step", "interrupt", "final").
        scaler: Optional AMP GradScaler instance.

    Returns:
        Path to the saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "loss": loss,
        "config": config_dict,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    filename = os.path.join(checkpoint_dir, f"pretrain_{tag}_{step}.pt")
    torch.save(checkpoint, filename)

    latest_path = os.path.join(checkpoint_dir, "pretrain_latest.pt")
    torch.save(checkpoint, latest_path)

    print(f"Checkpoint saved: {filename}")
    return filename


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    scaler: Optional[Any] = None,
) -> tuple[int, float]:
    """
    Load a training checkpoint and restore model/optimizer/scheduler/scaler state.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional LR scheduler to restore state.
        scaler: Optional AMP GradScaler to restore state.

    Returns:
        Tuple of (step, loss) from the checkpoint. If keys are missing,
        step defaults to 0 and loss to inf.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    step: int = checkpoint.get("step", 0)
    loss: float = checkpoint.get("loss", float("inf"))

    print(f"Checkpoint loaded: {checkpoint_path} (step={step}, loss={loss:.4f})")
    return step, loss


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the most recent checkpoint file in the given directory.

    Checks for 'pretrain_latest.pt' first (fast path), then falls back to
    scanning 'pretrain_step_*' files sorted by step number.

    Args:
        checkpoint_dir: Directory to scan for checkpoints.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found.
    """
    latest_path = os.path.join(checkpoint_dir, "pretrain_latest.pt")
    if os.path.exists(latest_path):
        return latest_path

    if not os.path.isdir(checkpoint_dir):
        return None

    try:
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("pretrain_step_") and f.endswith(".pt")
        ]
    except OSError:
        return None

    if not checkpoints:
        return None

    # Sort by step number extracted from filename: pretrain_step_{step}.pt
    def _extract_step(filename: str) -> int:
        # Strip prefix and suffix to isolate the step number
        core = filename[len("pretrain_step_"):]
        core = core.replace(".pt", "")
        try:
            return int(core)
        except ValueError:
            return 0

    checkpoints.sort(key=_extract_step)
    return os.path.join(checkpoint_dir, checkpoints[-1])


def save_best_checkpoint(
    model: nn.Module,
    step: int,
    loss: float,
    config: Any,
    checkpoint_dir: str,
) -> str:
    """
    Save the best model checkpoint (by validation loss).

    Overwrites any previous 'pretrain_best.pt'. Only saves model weights
    and metadata -- not optimizer/scheduler state.

    Args:
        model: The MiniLLM model.
        step: Training step at which this best result was achieved.
        loss: Validation loss value.
        config: Model config object.
        checkpoint_dir: Directory to save into.

    Returns:
        Path to the saved best checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    config_dict = config.to_dict() if hasattr(config, "to_dict") else config.__dict__

    best_path = os.path.join(checkpoint_dir, "pretrain_best.pt")
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
        "config": config_dict,
    }
    torch.save(checkpoint, best_path)
    print(f"Best checkpoint saved (loss={loss:.4f}) to: {best_path}")
    return best_path
