#!/usr/bin/env python3
# Derived from MIDI-VALLE and modified for VioLM by the VioLM authors, 2026.

"""Convert VioLM MIDI segments to articulation-aware token arrays."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from miditok import TokenizerConfig
from tqdm.auto import tqdm

from valle.utils.cli import ConsistentArgumentParser


BOS_ID = 1
EOS_ID = 2
MIDI_EXTENSIONS = {".mid", ".midi"}
logger = logging.getLogger("violm.midi_tokenize")


def _load_expression_tok(tokenizer_source: Path):
    """Load ``ExpressionTok`` from its historical hyphenated filename."""

    spec = importlib.util.spec_from_file_location("violm_tokenizer", tokenizer_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load tokenizer module from {tokenizer_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ExpressionTok


def _range_key_to_tuple(key: str) -> tuple[int, int]:
    start, end = key.split("_")
    return int(start), int(end)


def load_tokenizer_config(config_path: Path) -> TokenizerConfig:
    """Load the saved config while preserving MidiTok additional parameters."""

    data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    config_data = dict(data.get("config", data))
    if isinstance(config_data.get("beat_res"), dict):
        config_data["beat_res"] = {
            _range_key_to_tuple(key): value
            for key, value in config_data["beat_res"].items()
        }
    if isinstance(config_data.get("beat_res_rest"), dict):
        config_data["beat_res_rest"] = {
            _range_key_to_tuple(key): value
            for key, value in config_data["beat_res_rest"].items()
        }
    if isinstance(config_data.get("time_signature_range"), dict):
        config_data["time_signature_range"] = {
            int(key): value
            for key, value in config_data["time_signature_range"].items()
        }

    # MidiTok 3.x otherwise nests this mapping one level too deep when it is
    # passed as an unknown TokenizerConfig keyword.
    additional_params = dict(config_data.pop("additional_params", {}))
    additional_params.setdefault("data_type", "Midi")
    config = TokenizerConfig(**config_data)
    config.additional_params = additional_params
    return config


def create_tokenizer(config_path: Path, tokenizer_source: Path):
    expression_tok = _load_expression_tok(tokenizer_source)
    return expression_tok(load_tokenizer_config(config_path))


def tokenize_directory(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    tokenizer_source: Path,
    overwrite: bool = False,
    on_error: str = "fail",
) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"MIDI input directory does not exist: {input_dir}")
    tokenizer = create_tokenizer(config_path.resolve(), tokenizer_source.resolve())
    midi_files = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS
    )
    if not midi_files:
        raise FileNotFoundError(f"No .mid/.midi files found under {input_dir}")

    lengths: list[int] = []
    failures: list[tuple[Path, str]] = []
    written = 0
    skipped = 0
    for midi_path in tqdm(midi_files, desc="Tokenize MIDI"):
        relative = midi_path.relative_to(input_dir).with_suffix(".npy")
        output_path = output_dir / relative
        if output_path.exists() and not overwrite:
            skipped += 1
            continue
        try:
            sequences = tokenizer(midi_path)
            if not sequences or not sequences[0].ids:
                raise ValueError("Tokenizer returned an empty sequence")
            sequence = sequences[0].ids
            feature_count = len(sequence[0])
            if feature_count not in {6, 7}:
                raise ValueError(
                    f"Expected 6 or 7 MIDI feature streams, got {feature_count}"
                )
            tokens = np.asarray(
                [[BOS_ID] * feature_count, *sequence, [EOS_ID] * feature_count],
                dtype=np.int64,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, tokens)
            lengths.append(len(tokens))
            written += 1
        except Exception as exc:  # report every bad source without deleting data
            failures.append((midi_path, f"{type(exc).__name__}: {exc}"))
            logger.error("Failed to tokenize %s: %s", midi_path, failures[-1][1])
            if on_error == "fail":
                break

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files": len(midi_files),
        "written": written,
        "skipped": skipped,
        "failed": len(failures),
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "mean_length": float(np.mean(lengths)) if lengths else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tokenization_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if failures and on_error == "fail":
        details = "\n".join(f"- {path}: {error}" for path, error in failures[:10])
        raise RuntimeError(
            f"MIDI tokenization failed for {len(failures)} file(s):\n{details}"
        )
    if failures:
        logger.warning(
            "Skipped %d MIDI file(s); see tokenization_summary.json.",
            len(failures),
        )
    logger.info("MIDI tokenization summary: %s", summary)
    return summary


def get_parser() -> ConsistentArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = ConsistentArgumentParser(
        description="Tokenize .mid/.midi files for VioLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path,
        help="Root containing MIDI files; relative subdirectories are preserved.",
    )
    parser.add_argument("--output-dir", type=Path, help="Root for generated .npy files.")
    parser.add_argument(
        "--config", type=Path,
        default=repo_root / "tokenization" / "VioLM-tokenizer.json",
    )
    parser.add_argument(
        "--tokenizer-source", type=Path,
        default=repo_root / "tokenization" / "VioLM-tokenizer.py",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--on-error", choices=("fail", "skip"), default="fail")

    # Compatibility with the original MIDI-VALLE interface. Underscore
    # spellings are normalized by ConsistentArgumentParser as well.
    parser.add_argument("--data-folder", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--target-folder", type=Path, default=Path("midi_seg"), help=argparse.SUPPRESS)
    return parser


def _resolve_directories(args, parser) -> tuple[Path, Path]:
    input_dir = args.input_dir
    output_dir = args.output_dir
    if input_dir is None and args.data_folder is not None:
        input_dir = args.data_folder / args.target_folder
        if output_dir is None:
            output_dir = input_dir
        warnings.warn(
            "--data-folder/--target-folder are deprecated; use --input-dir and --output-dir.",
            FutureWarning,
            stacklevel=2,
        )
    if input_dir is None:
        parser.error("--input-dir is required")
    if output_dir is None:
        parser.error("--output-dir is required")
    return input_dir, output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    parser = get_parser()
    args = parser.parse_args()
    input_dir, output_dir = _resolve_directories(args, parser)
    tokenize_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=args.config,
        tokenizer_source=args.tokenizer_source,
        overwrite=args.overwrite,
        on_error=args.on_error,
    )


if __name__ == "__main__":
    main()
