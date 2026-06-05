"""
Logging utilities for MiniLLM training.

Provides:
- A standard Python logger setup with console (INFO) and file (DEBUG) handlers.
- A TensorBoard logger wrapper with graceful fallback when tensorboard is not installed.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str = "logs",
    prefix: str = "train",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Create and configure a logger with console and file handlers.

    The file handler writes to: ``{log_dir}/{prefix}_{timestamp}.log``.
    Returns an existing logger if one with the same name has already been configured.

    Args:
        name: Logger name (typically ``__name__``).
        log_dir: Directory in which to store log files.
        prefix: Prefix for the log file name.
        console_level: Logging level for console output (default INFO).
        file_level: Logging level for file output (default DEBUG).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is already configured.
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # --- Console handler (INFO and above) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # --- File handler (DEBUG and above) ---
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"{prefix}_{timestamp}.log")
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logging to file: {log_filename}")
    return logger


def get_logger(
    name: str,
    log_dir: str = "logs",
    prefix: str = "train",
) -> logging.Logger:
    """Convenience wrapper around :func:`setup_logger`.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        prefix: Prefix for the log file name.

    Returns:
        Configured logger.
    """
    return setup_logger(name, log_dir=log_dir, prefix=prefix)


# ---------------------------------------------------------------------------
# TensorBoard logger wrapper (graceful fallback)
# ---------------------------------------------------------------------------

_TENSORBOARD_AVAILABLE = False
try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    pass


class TBLogger:
    """Thin wrapper around PyTorch's ``SummaryWriter`` for TensorBoard logging.

    When ``tensorboard`` / ``torch.utils.tensorboard`` is not installed, all
    logging methods silently become no-ops so that training code does not need
    conditional branches.
    """

    def __init__(self, log_dir: str = "logs/tensorboard") -> None:
        """Create a TBLogger.

        If TensorBoard is available a ``SummaryWriter`` is created; otherwise
        the logger is a no-op and a warning is emitted once.

        Args:
            log_dir: Directory for TensorBoard event files.
        """
        self.log_dir = log_dir
        self.writer: Optional[object] = None
        self._warned: bool = False

        if _TENSORBOARD_AVAILABLE:
            os.makedirs(log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            if not self._warned:
                print(
                    "[TBLogger] tensorboard is not installed. "
                    "TensorBoard logging will be skipped."
                )
                self._warned = True

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric.

        Args:
            tag: Metric name (e.g. ``"train/loss"``).
            value: Scalar value.
            step: Global step counter.
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int) -> None:
        """Log multiple scalars under a shared prefix.

        Args:
            main_tag: Shared prefix tag.
            tag_scalar_dict: Dictionary mapping sub-tag names to values.
            step: Global step counter.
        """
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log a dictionary of metrics, each as a separate scalar.

        Common keys include: ``loss``, ``lr``, ``ppl``, ``grad_norm``.

        Args:
            metrics: Dictionary of {name: value} pairs.
            step: Global step counter.
        """
        if self.writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        """Flush and close the underlying writer (no-op if unavailable)."""
        if self.writer is not None:
            self.writer.close()
