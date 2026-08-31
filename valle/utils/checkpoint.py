"""Checkpoint helpers using a stable, plain-dictionary file format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel


def _unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def load_checkpoint(
    filename,
    model=None,
    model_avg=None,
    optimizer=None,
    scheduler=None,
) -> dict[str, Any]:
    checkpoint = torch.load(filename, map_location="cpu")
    if model is not None and "model" in checkpoint:
        _unwrap(model).load_state_dict(checkpoint["model"])
    if model_avg is not None and checkpoint.get("model_avg") is not None:
        _unwrap(model_avg).load_state_dict(checkpoint["model_avg"])
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def save_checkpoint(
    filename,
    model,
    params,
    model_avg=None,
    optimizer=None,
    scheduler=None,
    sampler=None,
    scaler=None,
    rank: int = 0,
) -> None:
    if rank != 0:
        return
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(params)
    payload.update(
        {
            "model": _unwrap(model).state_dict(),
            "model_avg": _unwrap(model_avg).state_dict() if model_avg is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "sampler": sampler.state_dict() if sampler is not None else None,
            "grad_scaler": scaler.state_dict() if scaler is not None else None,
        }
    )
    torch.save(payload, path)


def save_checkpoint_with_global_batch_idx(
    out_dir,
    global_batch_idx: int,
    **kwargs,
) -> None:
    save_checkpoint(Path(out_dir) / f"checkpoint-{global_batch_idx}.pt", **kwargs)


def remove_checkpoints(out_dir, topk: int, rank: int = 0) -> None:
    if rank != 0 or topk < 0:
        return
    paths = sorted(
        Path(out_dir).glob("checkpoint-*.pt"),
        key=lambda path: int(path.stem.rsplit("-", 1)[-1]),
        reverse=True,
    )
    for path in paths[topk:]:
        path.unlink()


@torch.no_grad()
def update_averaged_model(params, model_cur, model_avg) -> None:
    if model_avg is None:
        return
    model_cur = _unwrap(model_cur)
    weight = float(params.average_period) / float(params.batch_idx_train)
    for averaged, current in zip(model_avg.parameters(), model_cur.parameters()):
        averaged.mul_(1.0 - weight).add_(current.detach().to(averaged), alpha=weight)
