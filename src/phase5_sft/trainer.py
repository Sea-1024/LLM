"""
SFT Training script.

Loads a pretrained MiniLLM checkpoint, fine-tunes on instruction-following data,
and saves the best checkpoint based on validation loss.

Usage:
    python -m src.phase5_sft.trainer \
        --model_config configs/model/config_25m.yaml \
        --sft_config configs/sft/default.yaml \
        --pretrained_checkpoint models/checkpoints/pretrain_best.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# Project imports -- these modules are implemented elsewhere in the project.
from src.common.config import MiniLLMConfig, SFTConfig
from src.phase3_model.model import MiniLLM
from src.phase5_sft.sft_dataset import create_sft_dataloader
from src.phase5_sft.templates import PromptTemplate
from src.phase5_sft.data_format import format_and_tokenize


# ---------------------------------------------------------------------------
# LR scheduler: Cosine with linear warmup
# ---------------------------------------------------------------------------

def _cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create a cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Ratio of min_lr to peak_lr.

    Returns:
        A LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps:
            return min_lr_ratio
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1.0 + math.cos(math.pi * progress)
            )

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate(
    model: MiniLLM,
    val_loader: Any,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return loss and perplexity."""
    model.eval()
    total_loss: float = 0.0
    total_tokens: int = 0
    num_batches: int = 0

    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        # logits: (batch, seq_len, vocab_size)
        # labels: (batch, seq_len)  with -100 for masked positions

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )

        # Count non-masked tokens
        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens
        num_batches += 1

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))  # Cap to avoid overflow

    model.train()
    return {"loss": avg_loss, "ppl": ppl}


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: MiniLLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    loss: float,
    path: str,
    is_best: bool = False,
) -> None:
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler else {}
        ),
        "epoch": epoch,
        "step": step,
        "loss": loss,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "sft_best.pt")
        torch.save(checkpoint, best_path)
        print(f"[train_sft] Best checkpoint saved to {best_path}")


def _load_pretrained(
    model: MiniLLM,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    """Load pretrained weights into the model."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Remove unexpected keys (e.g. lm_head vs embedding weight tying)
    model_dict = model.state_dict()
    filtered_dict: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(filtered_dict, strict=False)
    if skipped:
        print(
            f"[train_sft] Skipped {len(skipped)} keys from pretrained "
            f"checkpoint (shape mismatch or not in model)"
        )
    if missing:
        print(
            f"[train_sft] {len(missing)} keys missing in checkpoint "
            f"(will be randomly initialized)"
        )


# ---------------------------------------------------------------------------
# Logger helper
# ---------------------------------------------------------------------------

class _Logger:
    """Simple logger that writes to stdout and optionally to a file."""

    def __init__(self, log_dir: str, name: str = "sft") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self.metrics_path = os.path.join(log_dir, f"{name}_metrics.jsonl")
        self.file_handle = open(self.log_path, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(line)
        self.file_handle.write(line + "\n")
        self.file_handle.flush()

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        metrics["timestamp"] = time.time()
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def close(self) -> None:
        self.file_handle.close()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_sft(
    model: MiniLLM,
    train_loader: Any,
    val_loader: Any,
    sft_config: SFTConfig,
    logger: _Logger,
) -> None:
    """
    Run the full SFT training loop.

    Features:
    - Gradient accumulation
    - Cosine LR schedule with warmup
    - Early stopping based on validation loss
    - Saves best and per-epoch checkpoints
    """
    device = next(model.parameters()).device

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    # Filter parameters that require grad
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=sft_config.learning_rate,
        weight_decay=sft_config.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ------------------------------------------------------------------
    # Scheduler: total steps
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader) // sft_config.gradient_accumulation_steps
    total_steps = steps_per_epoch * sft_config.epochs
    warmup_steps = int(total_steps * sft_config.warmup_ratio)

    scheduler = _cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=sft_config.min_lr / sft_config.learning_rate,
    )

    logger.log(f"Total training steps: {total_steps}")
    logger.log(f"Warmup steps: {warmup_steps}")
    logger.log(f"Gradient accumulation steps: {sft_config.gradient_accumulation_steps}")

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    no_improve_count: int = 0
    patience: int = sft_config.early_stop_patience

    checkpoint_dir = sft_config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # For tracking loss gap (overfitting indicator)
    train_loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(1, sft_config.epochs + 1):
        model.train()
        epoch_loss: float = 0.0
        epoch_tokens: int = 0
        accumulated_loss: float = 0.0
        step_in_epoch: int = 0

        t_start = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(input_ids)

            # Loss: only on non-masked tokens
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # Scale loss for gradient accumulation
            loss = loss / sft_config.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation step
            if (batch_idx + 1) % sft_config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), sft_config.max_grad_norm
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                step_in_epoch += 1

                # Accumulate epoch stats
                epoch_loss += accumulated_loss * sft_config.gradient_accumulation_steps
                accumulated_loss = 0.0

                # Logging
                if global_step % sft_config.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    current_loss = (
                        epoch_loss / max(step_in_epoch, 1)
                        if step_in_epoch > 0
                        else loss.item()
                    )
                    logger.log(
                        f"Epoch {epoch}/{sft_config.epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss {current_loss:.4f} | "
                        f"LR {lr:.2e}"
                    )

                # Validation
                if global_step % sft_config.eval_interval == 0:
                    val_metrics = _validate(model, val_loader, device)
                    train_loss = (
                        epoch_loss / max(step_in_epoch, 1)
                        if step_in_epoch > 0
                        else 0.0
                    )
                    loss_gap = train_loss - val_metrics["loss"]

                    logger.log(
                        f"[Eval] Step {global_step} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_metrics['loss']:.4f} | "
                        f"Val PPL: {val_metrics['ppl']:.2f} | "
                        f"Gap: {loss_gap:+.4f}"
                    )
                    logger.log_metrics({
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_ppl": val_metrics["ppl"],
                        "loss_gap": loss_gap,
                        "lr": scheduler.get_last_lr()[0],
                    })

        # ------------------------------------------------------------------
        # End of epoch
        # ------------------------------------------------------------------
        avg_epoch_loss = epoch_loss / max(step_in_epoch, 1)
        elapsed = time.time() - t_start

        # Validation
        val_metrics = _validate(model, val_loader, device)
        loss_gap = avg_epoch_loss - val_metrics["loss"]

        logger.log(
            f"=== Epoch {epoch} Complete === | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {avg_epoch_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val PPL: {val_metrics['ppl']:.2f} | "
            f"Gap: {loss_gap:+.4f}"
        )
        logger.log_metrics({
            "step": global_step,
            "epoch": epoch,
            "train_loss": avg_epoch_loss,
            "val_loss": val_metrics["loss"],
            "val_ppl": val_metrics["ppl"],
            "loss_gap": loss_gap,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_seconds": elapsed,
        })

        train_loss_history.append(avg_epoch_loss)

        # Save per-epoch checkpoint
        epoch_ckpt = os.path.join(checkpoint_dir, f"sft_epoch_{epoch}.pt")
        _save_checkpoint(
            model, optimizer, scheduler, epoch, global_step,
            avg_epoch_loss, epoch_ckpt, is_best=False,
        )

        # Check for improvement
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            no_improve_count = 0

            best_ckpt = os.path.join(checkpoint_dir, f"sft_epoch_{epoch}.pt")
            _save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                avg_epoch_loss, best_ckpt, is_best=True,
            )
        else:
            no_improve_count += 1
            logger.log(
                f"No improvement for {no_improve_count} epoch(s) "
                f"(best: {best_val_loss:.4f} at epoch {best_epoch})"
            )

            if no_improve_count >= patience:
                logger.log(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(patience={patience})"
                )
                break

    logger.log(
        f"Training finished. Best val loss: {best_val_loss:.4f} "
        f"at epoch {best_epoch}"
    )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised Fine-Tuning for MiniLLM"
    )
    parser.add_argument(
        "--model_config", type=str, required=True,
        help="Path to model config YAML."
    )
    parser.add_argument(
        "--sft_config", type=str, required=True,
        help="Path to SFT config YAML."
    )
    parser.add_argument(
        "--pretrained_checkpoint", type=str, required=True,
        help="Path to pretrained model checkpoint."
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', 'cpu', or 'cuda:0' etc."
    )
    parser.add_argument(
        "--skip_data_prep", action="store_true",
        help="Skip tokenization (use pre-tokenized binary data)."
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    model_config = MiniLLMConfig.from_yaml(args.model_config)
    sft_config = SFTConfig.from_yaml(args.sft_config)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[trainer] Using device: {device}")

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = _Logger(sft_config.log_dir, name="sft_train")

    try:
        # ------------------------------------------------------------------
        # Load tokenizer (from pretrained checkpoint directory)
        # ------------------------------------------------------------------
        # Tokenizer loading -- adjust based on how the tokenizer is saved.
        ckpt_dir = os.path.dirname(args.pretrained_checkpoint)
        tokenizer_path = os.path.join(ckpt_dir, "tokenizer.pkl")
        tokenizer = None
        if os.path.exists(tokenizer_path):
            import pickle
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)
            logger.log(f"Loaded tokenizer from {tokenizer_path}")
        else:
            logger.log(
                "[trainer] WARNING: tokenizer not found, will try to load "
                "from model attribute."
            )

        # ------------------------------------------------------------------
        # Prepare data (tokenize if needed)
        # ------------------------------------------------------------------
        template = PromptTemplate(sft_config.template_type)

        train_bin_dir = os.path.join(
            os.path.dirname(sft_config.train_data_path), "binary", "train"
        )
        val_bin_dir = os.path.join(
            os.path.dirname(sft_config.val_data_path), "binary", "val"
        )

        if not args.skip_data_prep:
            if tokenizer is None:
                raise RuntimeError(
                    "Tokenizer is required for data preparation. "
                    "Provide --tokenizer_path or place tokenizer.pkl "
                    "next to the checkpoint."
                )

            logger.log("Tokenizing training data...")
            format_and_tokenize(
                data_path=sft_config.train_data_path,
                tokenizer=tokenizer,
                template=template,
                output_dir=train_bin_dir,
                max_seq_len=sft_config.max_seq_len,
            )

            logger.log("Tokenizing validation data...")
            format_and_tokenize(
                data_path=sft_config.val_data_path,
                tokenizer=tokenizer,
                template=template,
                output_dir=val_bin_dir,
                max_seq_len=sft_config.max_seq_len,
            )

        train_data_path = os.path.join(train_bin_dir, "data.bin")
        train_prompt_len_path = os.path.join(train_bin_dir, "prompt_lens.npy")
        val_data_path = os.path.join(val_bin_dir, "data.bin")
        val_prompt_len_path = os.path.join(val_bin_dir, "prompt_lens.npy")

        # ------------------------------------------------------------------
        # Build model
        # ------------------------------------------------------------------
        logger.log(f"Building model from config: {args.model_config}")
        model = MiniLLM(model_config)
        model.to(device)

        # Apply SFT dropout (may differ from pretrain)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = sft_config.dropout

        # ------------------------------------------------------------------
        # Load pretrained weights
        # ------------------------------------------------------------------
        logger.log(f"Loading pretrained checkpoint: {args.pretrained_checkpoint}")
        _load_pretrained(model, args.pretrained_checkpoint, device)
        logger.log("Pretrained weights loaded successfully.")

        # ------------------------------------------------------------------
        # Data loaders
        # ------------------------------------------------------------------
        logger.log("Creating data loaders...")
        train_loader = create_sft_dataloader(
            data_path=train_data_path,
            batch_size=sft_config.batch_size,
            shuffle=True,
            prompt_len_path=train_prompt_len_path,
            num_workers=0,
            seq_len=sft_config.max_seq_len,
            drop_last=True,
        )
        val_loader = create_sft_dataloader(
            data_path=val_data_path,
            batch_size=sft_config.batch_size,
            shuffle=False,
            prompt_len_path=val_prompt_len_path,
            num_workers=0,
            seq_len=sft_config.max_seq_len,
            drop_last=False,
        )
        logger.log(
            f"Train batches: {len(train_loader)}, "
            f"Val batches: {len(val_loader)}"
        )

        # ------------------------------------------------------------------
        # Train
        # ------------------------------------------------------------------
        logger.log("Starting SFT training...")
        train_sft(model, train_loader, val_loader, sft_config, logger)

        # ------------------------------------------------------------------
        # Save final model
        # ------------------------------------------------------------------
        final_path = os.path.join(sft_config.checkpoint_dir, "sft_final.pt")
        torch.save({"model_state_dict": model.state_dict()}, final_path)
        logger.log(f"Final model saved to {final_path}")

    except Exception as e:
        logger.log(f"ERROR: {e}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
