"""
Phase 4: Pretraining Trainer.

Main entry point for pretraining the MiniLLM model from scratch or from a checkpoint.

Usage:
    python -m src.phase4_pretrain.trainer \
        --model_config configs/model/config_25m.yaml \
        --pretrain_config configs/pretrain/default.yaml

Supports: gradient accumulation, warmup + cosine LR schedule, mixed precision (AMP),
checkpoint resume, periodic evaluation, and graceful interrupt handling.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project imports -- these modules must exist before running this script
from src.common.config import MiniLLMConfig, PretrainConfig
from src.common.utils import set_seed, setup_logging
from src.phase3_model.model import MiniLLM
from src.phase1_data.dataset import create_pretrain_datasets
from src.phase4_pretrain.metrics import (
    AverageMeter,
    compute_grad_norm,
    compute_perplexity,
    evaluate_model,
)
from src.phase4_pretrain.checkpoint import (
    get_latest_checkpoint,
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# YAML config helper
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML configuration file. Requires PyYAML."""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to load config files. Install with: pip install pyyaml"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")
    return data


# ---------------------------------------------------------------------------
# LR schedule helper: warmup (linear) + cosine decay
# ---------------------------------------------------------------------------

def _build_lr_lambda(
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
):
    """
    Return a lambda for torch LambdaLR that implements:
      - Linear warmup from 0 to peak_lr over ``warmup_steps``.
      - Cosine annealing from peak_lr to min_lr over remaining steps.

    The lambda returns the **multiplier** applied to the base LR set in the
    optimizer.  So it ranges from 0 -> 1 during warmup, then from 1 down to
    (min_lr / peak_lr) during cosine decay.
    """
    safe_warmup = max(warmup_steps, 1)
    safe_total = max(total_steps, 1)
    min_ratio = min_lr / peak_lr if peak_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # linear warmup: 0 -> 1
            return step / safe_warmup
        # cosine decay: 1 -> min_ratio
        progress = (step - warmup_steps) / max(safe_total - warmup_steps, 1)
        progress = min(progress, 1.0)  # clamp in case step >= total_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


# ---------------------------------------------------------------------------
# AMP context helper
# ---------------------------------------------------------------------------

def _get_amp_context(use_amp: bool, device: torch.device):
    """
    Return a no-op or autocast context manager suitable for the device.
    On CPU: uses torch.cpu.amp.autocast (bf16, no gradient scaling needed).
    On CUDA: uses torch.cuda.amp.autocast.
    """
    if not use_amp:
        from contextlib import nullcontext
        return nullcontext()

    if device.type == "cuda":
        return torch.cuda.amp.autocast()
    else:
        # CPU path: bfloat16 autocast (PyTorch >= 1.10)
        return torch.cpu.amp.autocast(dtype=torch.bfloat16)


def _get_scaler(use_amp: bool, device: torch.device) -> Optional[Any]:
    """
    Return a GradScaler if using CUDA AMP, otherwise None.
    CPU bf16 autocast does not require gradient scaling.
    """
    if use_amp and device.type == "cuda":
        return torch.cuda.amp.GradScaler()
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain MiniLLM from scratch or resume from checkpoint."
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model config YAML (e.g. configs/model/config_25m.yaml).",
    )
    parser.add_argument(
        "--pretrain_config",
        type=str,
        required=True,
        help="Path to pretrain config YAML (e.g. configs/pretrain/default.yaml).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume from (overrides auto-latest).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="models/tokenizer/tokenizer.json",
        help="Path to the trained HuggingFace tokenizer JSON file.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Load configurations -------------------------------------------------
    model_cfg_dict = _load_yaml(args.model_config)
    pretrain_cfg_dict = _load_yaml(args.pretrain_config)

    model_config = MiniLLMConfig(**model_cfg_dict)
    pretrain_config = PretrainConfig(**pretrain_cfg_dict)

    # ---- Setup logging -------------------------------------------------------
    os.makedirs(pretrain_config.log_dir, exist_ok=True)
    os.makedirs(pretrain_config.checkpoint_dir, exist_ok=True)

    logger = setup_logging("pretrain", pretrain_config.log_dir)

    logger.info("=" * 60)
    logger.info("Phase 4: Pretraining")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Pretrain config: {args.pretrain_config}")
    logger.info("=" * 60)

    # ---- Reproducibility -----------------------------------------------------
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    # ---- Device --------------------------------------------------------------
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # ---- Model ---------------------------------------------------------------
    logger.info("Creating model...")
    model = MiniLLM(model_config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Print model config summary
    logger.info(
        f"Model config: d_model={model_config.d_model}, "
        f"n_layers={model_config.n_layers}, "
        f"n_heads={model_config.n_heads}, "
        f"d_ff={model_config.d_ff}, "
        f"vocab_size={model_config.vocab_size}, "
        f"max_seq_len={model_config.max_seq_len}"
    )

    # ---- Tokenizer -----------------------------------------------------------
    logger.info("Loading tokenizer...")
    tokenizer_path = args.tokenizer_path
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Please train the tokenizer first (phase 2)."
        )

    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        logger.info(f"Tokenizer loaded, vocab size: {tokenizer.get_vocab_size()}")
    except ImportError:
        raise ImportError(
            "HuggingFace 'tokenizers' library is required. "
            "Install with: pip install tokenizers"
        )

    # ---- Datasets & DataLoaders ----------------------------------------------
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_pretrain_datasets(
        train_data_path=pretrain_config.train_data_path,
        val_data_path=pretrain_config.val_data_path,
        max_seq_len=model_config.max_seq_len,
    )
    logger.info(f"Train dataset size: {len(train_dataset)} samples")
    logger.info(f"Val dataset size: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=pretrain_config.batch_size,
        shuffle=True,
        num_workers=0,            # CPU training: single-process is simpler / safer
        pin_memory=False,
        drop_last=True,            # avoid tiny last batch messing up batch-norm stats
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=pretrain_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # ---- Optimizer -----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pretrain_config.learning_rate,
        betas=pretrain_config.betas,
        eps=pretrain_config.eps,
        weight_decay=pretrain_config.weight_decay,
    )
    logger.info(
        f"Optimizer: AdamW (lr={pretrain_config.learning_rate}, "
        f"betas={pretrain_config.betas}, eps={pretrain_config.eps}, "
        f"weight_decay={pretrain_config.weight_decay})"
    )

    # ---- LR Scheduler: warmup + cosine ---------------------------------------
    lr_lambda_fn = _build_lr_lambda(
        warmup_steps=pretrain_config.warmup_steps,
        total_steps=pretrain_config.max_steps,
        peak_lr=pretrain_config.learning_rate,
        min_lr=pretrain_config.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)
    logger.info(
        f"Scheduler: linear warmup ({pretrain_config.warmup_steps} steps) "
        f"+ cosine decay to {pretrain_config.min_lr:.2e}"
    )

    # ---- AMP (mixed precision) -----------------------------------------------
    use_amp = pretrain_config.use_amp
    amp_context = _get_amp_context(use_amp, device)
    scaler = _get_scaler(use_amp, device)
    if use_amp:
        amp_type = "CUDA fp16" if device.type == "cuda" else "CPU bf16"
        logger.info(f"AMP enabled: {amp_type}")
    else:
        logger.info("AMP disabled (full precision)")

    # ---- Resume from checkpoint ----------------------------------------------
    start_step = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        logger.info(f"Resuming from user-specified checkpoint: {args.resume}")
        start_step, _ = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        # step in checkpoint is the step that was just completed; resume from next
        start_step += 1
    else:
        latest_ckpt = get_latest_checkpoint(pretrain_config.checkpoint_dir)
        if latest_ckpt is not None:
            logger.info(f"Auto-resuming from latest checkpoint: {latest_ckpt}")
            start_step, _ = load_checkpoint(
                latest_ckpt, model, optimizer, scheduler, scaler
            )
            start_step += 1

    if start_step > 0:
        # Run a quick evaluation to establish the current best_val_loss
        logger.info("Running quick evaluation to determine best val loss so far...")
        current_val_loss, current_val_ppl = evaluate_model(
            model, val_loader, device, max_batches=50
        )
        best_val_loss = current_val_loss
        logger.info(
            f"Resumed model: step={start_step}, val_loss={current_val_loss:.4f}, "
            f"val_ppl={current_val_ppl:.2f}"
        )

    # ---- Training state -------------------------------------------------------
    model.train()
    grad_accum_steps = pretrain_config.gradient_accumulation_steps
    max_steps = pretrain_config.max_steps

    logger.info(f"Training from step {start_step} to {max_steps}")
    logger.info(f"Micro batch size: {pretrain_config.batch_size}")
    logger.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logger.info(
        f"Effective batch size: {pretrain_config.batch_size * grad_accum_steps}"
    )

    # Metrics tracking
    loss_meter = AverageMeter()
    total_tokens = 0
    training_start_time = time.time()
    step_time = 0.0  # seconds for the most recent step

    optimizer.zero_grad()

    # ---- Main training loop --------------------------------------------------
    train_iter = iter(train_loader)

    try:
        for step in range(start_step, max_steps):
            step_start = time.time()

            # --- Fetch batch (with automatic epoch cycling) -------------------
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids: torch.Tensor = batch[0].to(device)
            labels: torch.Tensor = batch[1].to(device)
            batch_tokens = input_ids.numel()

            # --- Forward pass -------------------------------------------------
            with amp_context:
                logits = model(input_ids)  # (B, S, vocab_size)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=model_config.pad_token_id,
                )
                # Normalize loss by accumulation steps so the sum of
                # micro-batch losses equals the effective-batch loss.
                loss = loss / grad_accum_steps

            # --- Backward pass ------------------------------------------------
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Record unscaled loss for logging
            loss_meter.update(loss.item() * grad_accum_steps, n=batch_tokens)
            total_tokens += batch_tokens

            # --- Gradient accumulation & optimizer step -----------------------
            # Step optimizer and scheduler every grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                # Gradient clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), pretrain_config.max_grad_norm
                )

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            # --- Logging ------------------------------------------------------
            if (step + 1) % pretrain_config.log_interval == 0:
                elapsed = time.time() - training_start_time
                tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
                current_lr = scheduler.get_last_lr()[0]
                ppl = compute_perplexity(loss_meter.avg)

                logger.info(
                    f"Step {step + 1:>6d}/{max_steps} | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"PPL: {ppl:.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Tokens: {total_tokens:,} | "
                    f"Step time: {time.time() - step_start:.2f}s"
                )

                # Also compute and log gradient norm (useful for debugging)
                if (step + 1) % (pretrain_config.log_interval * 5) == 0:
                    gnorm = compute_grad_norm(model)
                    logger.info(f"  -> Gradient norm: {gnorm:.4f}")

                loss_meter.reset()

            # --- Evaluation ---------------------------------------------------
            if (step + 1) % pretrain_config.eval_interval == 0:
                logger.info(f"Running evaluation at step {step + 1}...")
                val_loss, val_ppl = evaluate_model(
                    model, val_loader, device, max_batches=None
                )
                logger.info(
                    f"  Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_best_checkpoint(
                        model,
                        step + 1,
                        val_loss,
                        model_config,
                        pretrain_config.checkpoint_dir,
                    )
                    logger.info(f"  New best val loss: {best_val_loss:.4f}")

            # --- Checkpoint save ----------------------------------------------
            if (step + 1) % pretrain_config.save_interval == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step + 1,
                    loss_meter.avg,
                    model_config,
                    pretrain_config.checkpoint_dir,
                    tag="step",
                    scaler=scaler,
                )

            step_time = time.time() - step_start

    except KeyboardInterrupt:
        logger.info(
            f"\nTraining interrupted by user at step {step + 1}. "
            "Saving checkpoint before exit..."
        )
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            step + 1,
            loss_meter.avg,
            model_config,
            pretrain_config.checkpoint_dir,
            tag="interrupt",
            scaler=scaler,
        )
        logger.info("Checkpoint saved. Exiting.")
        sys.exit(0)

    # ---- Training complete ---------------------------------------------------
    total_time = time.time() - training_start_time

    # Final checkpoint
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        max_steps,
        loss_meter.avg,
        model_config,
        pretrain_config.checkpoint_dir,
        tag="final",
        scaler=scaler,
    )

    # Final evaluation
    logger.info("Running final evaluation...")
    final_val_loss, final_val_ppl = evaluate_model(
        model, val_loader, device, max_batches=None
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Total steps:            {max_steps}")
    logger.info(f"  Total tokens processed: {total_tokens:,}")
    logger.info(f"  Total training time:    {total_time / 3600:.2f} hours ({total_time:.0f}s)")
    logger.info(f"  Avg tokens/sec:         {total_tokens / total_time:.0f}")
    logger.info(f"  Final val loss:         {final_val_loss:.4f}")
    logger.info(f"  Final val perplexity:   {final_val_ppl:.2f}")
    logger.info(f"  Best val loss:          {best_val_loss:.4f}")
    logger.info(f"  Best val perplexity:    {compute_perplexity(best_val_loss):.2f}")
    logger.info("=" * 60)

    # Sanity check: if best is still inf we never evaluated
    if best_val_loss == float("inf"):
        logger.warning(
            "Best validation loss was not updated. "
            "Consider reducing eval_interval so evaluation runs at least once."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
