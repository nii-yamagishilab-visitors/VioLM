#!/usr/bin/env python3
# Derived from MIDI-VALLE and modified for VioLM by the VioLM authors, 2026.

"""Synthesize violin audio from tokenized prompt and target MIDI."""

from __future__ import annotations

import argparse
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torchaudio

from valle.data import AudioTokenizer32, AudioTokenizer32FT, tokenize_audio
from valle.data.collation import get_midi_token_collater
from valle.models import get_model
from valle.models.lora_valle import (
    _parse_targets,
    apply_lora_to_valle,
    use_attn_qkv_linear,
    use_attn_separate_qkv_linear,
)
from valle.utils.cli import ConsistentArgumentParser
from valle.utils.runtime import AttributeDict, select_device, str2bool


logger = logging.getLogger("violm.infer")


def _expand_paths(values: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        paths.extend(Path(part) for part in value.split("|") if part)
    return paths


def _load_tokens(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"MIDI token file does not exist: {path}")
    tokens = np.load(path, allow_pickle=False)
    if tokens.ndim != 2 or tokens.shape[1] not in {6, 7}:
        raise ValueError(
            f"Expected a [notes, 6|7] token array in {path}, got {tokens.shape}"
        )
    return tokens.astype(np.int64, copy=False)


def _concat_token_files(paths: list[Path]) -> np.ndarray:
    arrays = [_load_tokens(path) for path in paths]
    if not arrays:
        raise ValueError("At least one MIDI token file is required")
    feature_count = arrays[0].shape[1]
    if any(array.shape[1] != feature_count for array in arrays):
        raise ValueError("All MIDI token files must use the same number of streams")
    combined = arrays[0]
    for array in arrays[1:]:
        combined = np.concatenate([combined[:-1], array[1:]], axis=0)
    return combined


def _lora_targets_from_state(state: dict[str, torch.Tensor]) -> str:
    targets = set()
    keys = state.keys()
    if any(".linear1.lora_" in key or ".linear2.lora_" in key for key in keys):
        targets.add("ffn")
    if any(".out_proj.lora_" in key for key in keys):
        targets.add("attn_out")
    for name in ("q", "k", "v"):
        if any(f".{name}_proj.lora_" in key for key in keys):
            targets.add(f"attn_{name}")
    if any(".in_proj_linear.lora_" in key for key in keys):
        targets.add("attn_qkv")
    return "+".join(sorted(targets))


def _first_state_tensor(state, suffix: str):
    return next((value for key, value in state.items() if key.endswith(suffix)), None)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    use_lora: Optional[bool] = None,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    lora_target_modules: Optional[str] = None,
):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)

    text_projection = _first_state_tensor(state, "ar_text_embedding.weight")
    if text_projection is None:
        raise KeyError("Checkpoint has no ar_text_embedding.weight")
    articulation = text_projection.shape[1] == 560

    detected_lora = any("lora_" in key for key in state)
    use_lora = bool(checkpoint.get("use_lora", detected_lora)) if use_lora is None else use_lora
    if use_lora != detected_lora:
        raise ValueError(
            f"Checkpoint LoRA detection is {detected_lora}, but --use-lora resolved to {use_lora}"
        )
    if use_lora:
        if lora_rank is None:
            lora_rank = checkpoint.get("lora_rank")
        if lora_rank is None:
            lora_a = _first_state_tensor(state, "lora_A")
            lora_rank = int(lora_a.shape[0]) if lora_a is not None else 8
        lora_alpha = int(lora_alpha if lora_alpha is not None else checkpoint.get("lora_alpha", 16))
        lora_dropout = float(lora_dropout if lora_dropout is not None else checkpoint.get("lora_dropout", 0.0))
        lora_target_modules = lora_target_modules or checkpoint.get("lora_target_modules") or _lora_targets_from_state(state)
        if not lora_target_modules:
            raise ValueError("Could not infer LoRA target modules from the checkpoint")

    params = AttributeDict(
        model_name=checkpoint.get("model_name", "valle"),
        decoder_dim=int(checkpoint.get("decoder_dim", text_projection.shape[0])),
        nhead=int(checkpoint.get("nhead", 16)),
        num_decoder_layers=int(checkpoint.get("num_decoder_layers", 12)),
        norm_first=bool(checkpoint.get("norm_first", True)),
        add_prenet=bool(checkpoint.get("add_prenet", False)),
        prefix_mode=int(checkpoint.get("prefix_mode", 1)),
        share_embedding=bool(checkpoint.get("share_embedding", True)),
        scale_factor=float(checkpoint.get("scale_factor", 1.0)),
        prepend_bos=bool(checkpoint.get("prepend_bos", False)),
        num_quantizers=int(checkpoint.get("num_quantizers", 4)),
        articulation=articulation,
        scaling_xformers=bool(checkpoint.get("scaling_xformers", False)),
    )

    if use_lora:
        targets = _parse_targets(lora_target_modules)
        separate_qkv = bool(targets & {"attn_q", "attn_k", "attn_v"})
        fused_qkv = "attn_qkv" in targets
        if separate_qkv and fused_qkv:
            raise ValueError("A LoRA checkpoint cannot mix fused and separate Q/K/V targets")
        build_context = (
            use_attn_separate_qkv_linear()
            if separate_qkv
            else use_attn_qkv_linear()
            if fused_qkv
            else nullcontext()
        )
    else:
        build_context = nullcontext()

    with build_context:
        model = get_model(params)
    if use_lora:
        apply_lora_to_valle(
            model,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
            freeze_base=False,
        )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    logger.info(
        "Loaded %s (articulation=%s, LoRA=%s, decoder_dim=%d, layers=%d)",
        checkpoint_path,
        articulation,
        use_lora,
        params.decoder_dim,
        params.num_decoder_layers,
    )
    return model


def _create_audio_tokenizer(name: str, checkpoint: Path | None, device):
    normalized = name.lower().replace("_", "-")
    if normalized in {"encodec32-ft", "encodec32ft"}:
        return AudioTokenizer32FT(device=device, checkpoint=checkpoint)
    if normalized in {"encodec32", "encodec-32"}:
        return AudioTokenizer32(device=device)
    raise ValueError("--audio-tokenizer must be encodec32-ft or encodec32")


def _output_path(output_dir: Path, output_file: str, index: int, total: int) -> Path:
    requested = Path(output_file)
    suffix = requested.suffix or ".wav"
    stem = requested.stem
    name = f"{stem}{suffix}" if total == 1 else f"{stem}-{index:03d}{suffix}"
    return output_dir / name


def get_parser() -> ConsistentArgumentParser:
    parser = ConsistentArgumentParser(
        description="Synthesize violin audio from VioLM MIDI tokens",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path)
    parser.add_argument("--midi-prompts", nargs="+", required=True, help="Prompt .npy files; '|' input remains supported.")
    parser.add_argument("--audio-prompts", nargs="+", required=True, help="Aligned prompt audio files.")
    parser.add_argument("--midi-tokens", "--midi", dest="midi_tokens", nargs="+", required=True, help="Target .npy files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-file", default="violm.wav")
    parser.add_argument("--audio-tokenizer", default="encodec32-ft")
    parser.add_argument("--top-k", type=int, default=-100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continual", type=str2bool, default=False)
    parser.add_argument("--use-lora", type=str2bool, default=None, help="Normally auto-detected from checkpoint.")
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--lora-target-modules")
    parser.add_argument(
        "--device",
        default="auto",
        help="For example cuda:1 or cpu; auto selects the GPU with most free memory.",
    )
    return parser


@torch.no_grad()
def run(args: argparse.Namespace) -> list[Path]:
    device = select_device(args.device)
    logger.info("Inference device: %s", device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    midi_prompt_paths = _expand_paths(args.midi_prompts)
    audio_prompt_paths = _expand_paths(args.audio_prompts)
    target_paths = _expand_paths(args.midi_tokens)
    if len(midi_prompt_paths) != len(audio_prompt_paths):
        raise ValueError("--midi-prompts and --audio-prompts must contain the same number of files")
    prompt_tokens = _concat_token_files(midi_prompt_paths)

    model = load_model(
        args.checkpoint.resolve(),
        device,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    audio_tokenizer = _create_audio_tokenizer(
        args.audio_tokenizer,
        args.codec_checkpoint.resolve() if args.codec_checkpoint else None,
        device,
    )
    prompt_frames = [
        tokenize_audio(args.audio_tokenizer, str(path), audio_tokenizer)
        for path in audio_prompt_paths
    ]
    audio_prompts = torch.cat(prompt_frames, dim=-1).transpose(2, 1).to(device)
    midi_collater = get_midi_token_collater()
    _, enroll_x_lens = midi_collater([prompt_tokens[:-1].tolist()])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, target_path in enumerate(target_paths):
        output_path = _output_path(args.output_dir, args.output_file, index, len(target_paths))
        if output_path.exists() and not args.overwrite:
            logger.info("Skipping existing output: %s", output_path)
            outputs.append(output_path)
            continue
        target_tokens = _load_tokens(target_path)
        if target_tokens.shape[1] != prompt_tokens.shape[1]:
            raise ValueError(
                f"Prompt has {prompt_tokens.shape[1]} streams but {target_path} has {target_tokens.shape[1]}"
            )
        conditioned_tokens = np.concatenate(
            [prompt_tokens[:-1], target_tokens[1:]], axis=0
        )
        midi_tokens, midi_lengths = midi_collater([conditioned_tokens.tolist()])
        if args.continual:
            encoded_frames = model.continual(
                midi_tokens.to(device), midi_lengths.to(device), audio_prompts
            )
        else:
            encoded_frames = model.inference(
                midi_tokens.to(device),
                midi_lengths.to(device),
                audio_prompts,
                enroll_x_lens=enroll_x_lens.to(device),
                top_k=args.top_k,
                temperature=args.temperature,
            )
        samples = audio_tokenizer.decode(encoded_frames.transpose(2, 1), None)
        torchaudio.save(str(output_path), samples[0].cpu(), audio_tokenizer.sample_rate)
        outputs.append(output_path)
        logger.info("Wrote %s", output_path)
    return outputs


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
