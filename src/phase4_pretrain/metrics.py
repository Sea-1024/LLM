"""
Training metrics utilities for pretraining.
Provides loss computation, perplexity, gradient norm, evaluation, and running average tracking.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.

    PPL = exp(loss), clamped at exp(20) to avoid float overflow
    for very high loss values (e.g. at initialization).
    """
    clamped_loss = min(loss, 20.0)
    return math.exp(clamped_loss)


def compute_grad_norm(model: nn.Module) -> float:
    """
    Compute total gradient L2 norm across all parameters with gradients.

    Returns 0.0 if no parameters have gradients.
    """
    total_norm = 0.0
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            has_grad = True
    if not has_grad:
        return 0.0
    return total_norm ** 0.5


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> tuple[float, float]:
    """
    Evaluate model on validation set.

    Runs the model in eval mode (no dropout), computes average
    cross-entropy loss and perplexity over the validation set.

    Args:
        model: The MiniLLM model instance.
        val_loader: DataLoader for validation data, yielding (input_ids, labels) batches.
        device: torch device to run evaluation on.
        max_batches: If set, limit evaluation to this many batches (faster).

    Returns:
        Tuple of (average_loss, perplexity).
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pad_token_id = model.config.pad_token_id

    for batch in val_loader:
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=pad_token_id,
        )

        total_loss += loss.item()
        num_batches += 1

        if max_batches is not None and num_batches >= max_batches:
            break

    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    ppl = compute_perplexity(avg_loss)
    return avg_loss, ppl


class AverageMeter:
    """
    Tracks running average and current value of a scalar metric.

    Useful for tracking loss, throughput, etc. across training steps.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated values."""
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Update the meter with a new value.

        Args:
            val: The new value to record.
            n: The weight/count for this value (e.g. batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
