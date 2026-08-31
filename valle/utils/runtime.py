"""Self-contained runtime helpers used by VioLM.

MIDI-VALLE imported these helpers from a source checkout of ``icefall``. That
made installation depend on a matching k2 wheel and a manually configured
``PYTHONPATH`` even though VioLM only needs a few generic utilities.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch


class AttributeDict(dict):
    """Dictionary whose keys are also accessible as attributes."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def str2bool(value: Any) -> bool:
    """Parse common command-line boolean spellings."""

    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Return ``True`` at padded positions for sequence lengths."""

    if lengths.ndim != 1:
        raise ValueError(f"lengths must be one-dimensional, got {lengths.shape}")
    if max_len is None:
        max_len = int(lengths.max().item()) if lengths.numel() else 0
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) >= lengths.unsqueeze(1)


def select_device(requested: str | None = None) -> torch.device:
    """Resolve a device, choosing the CUDA GPU with most free memory for auto."""

    if requested and requested.lower() != "auto":
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_memory = []
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(index)
        except RuntimeError:
            free_bytes = 0
        free_memory.append(free_bytes)
    best_index = max(range(len(free_memory)), key=free_memory.__getitem__)
    return torch.device("cuda", best_index)


class MetricsTracker(defaultdict):
    """Numeric metric accumulator compatible with the training loop."""

    def __init__(self, *args, **kwargs):
        super().__init__(float)
        self.update(*args, **kwargs)

    def _combine(self, other: "MetricsTracker", operation) -> "MetricsTracker":
        result = MetricsTracker()
        for key in self.keys() | other.keys():
            result[key] = operation(self[key], other[key])
        return result

    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        return self._combine(other, lambda left, right: left + right)

    def __mul__(self, scale: float) -> "MetricsTracker":
        return MetricsTracker({key: value * scale for key, value in self.items()})

    __rmul__ = __mul__

    def reduce(self, device: torch.device | str) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        keys = sorted(self)
        values = torch.tensor([self[key] for key in keys], device=device)
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        for key, value in zip(keys, values.tolist()):
            self[key] = value

    def write_summary(self, writer, prefix: str, step: int) -> None:
        frames = max(float(self.get("frames", 0.0)), 1.0)
        for key, value in self.items():
            summary_value = value if key in {"frames", "utterances"} else value / frames
            writer.add_scalar(f"{prefix}{key}", summary_value, step)

    def __str__(self) -> str:
        frames = max(float(self.get("frames", 0.0)), 1.0)
        fields = []
        for key in sorted(self):
            value = self[key] if key in {"frames", "utterances"} else self[key] / frames
            fields.append(f"{key}={value:.4g}")
        return ", ".join(fields)


def setup_logger(log_filename: str | os.PathLike[str]) -> None:
    """Configure console and UTF-8 file logging."""

    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_env_info() -> dict[str, str]:
    """Collect compact environment metadata saved in checkpoints."""

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
        "lhotse": version("lhotse"),
        "violm": version("violm-ismir26"),
    }


def register_inf_check_hooks(model: torch.nn.Module) -> None:
    """Raise immediately when a module emits NaN or infinite tensors."""

    def check_output(module, _inputs, output):
        tensors: Iterator[torch.Tensor]
        if isinstance(output, torch.Tensor):
            tensors = iter((output,))
        elif isinstance(output, (tuple, list)):
            tensors = (item for item in output if isinstance(item, torch.Tensor))
        else:
            return
        for tensor in tensors:
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                raise FloatingPointError(
                    f"Non-finite output detected in {module.__class__.__name__}"
                )

    for module in model.modules():
        module.register_forward_hook(check_output)


@contextlib.contextmanager
def get_executor():
    """Yield no external executor; Lhotse will use local worker processes."""

    yield None


def setup_dist(rank: int, world_size: int, master_port: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_dist() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
