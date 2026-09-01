#!/usr/bin/env python3
# Derived from MIDI-VALLE and modified for VioLM by the VioLM authors, 2026.

"""Extract audio codec features and assemble final VioLM CutSet manifests."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
from lhotse import CutSet, NumpyHdf5Writer
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm.auto import tqdm

from valle.data import (
    AudioTokenConfig,
    AudioTokenConfig32,
    AudioTokenExtractor,
    AudioTokenExtractor32,
    AudioTokenExtractor32FT,
)
from valle.data.fbank import get_fbank_extractor
from valle.utils.cli import ConsistentArgumentParser
from valle.utils.runtime import get_executor
from valle.utils.runtime import select_device


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")
logger = logging.getLogger("violm.audio_tokenize")


def normalize_extractor(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "encodec": "encodec",
        "encodec32": "encodec32",
        "encodec-32": "encodec32",
        "encodec32ft": "encodec32-ft",
        "encodec32-ft": "encodec32-ft",
        "fbank": "fbank",
    }
    if normalized not in aliases:
        raise argparse.ArgumentTypeError(
            f"Unknown audio extractor {value!r}; choose encodec, encodec32, encodec32-ft, or fbank"
        )
    return aliases[normalized]


def get_parser() -> ConsistentArgumentParser:
    parser = ConsistentArgumentParser(
        description="Extract audio tokens and create cuts_*.jsonl.gz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest-dir", "--src-dir", dest="manifest_dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-extractor", type=normalize_extractor, default="encodec32-ft")
    parser.add_argument(
        "--codec-checkpoint", type=Path,
        help="Required by encodec32-ft; the hybrid EnCodec .bin file.",
    )
    parser.add_argument("--dataset-parts", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--prefix", default="", help="Optional manifest/file prefix without a trailing underscore.")
    parser.add_argument("--suffix", default="jsonl.gz")
    parser.add_argument("--batch-duration", type=float, default=100.0)
    parser.add_argument("--num-jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        help="Audio codec device; auto selects the CUDA GPU with most free memory.",
    )
    return parser


def _normalize_parts(values: list[str]) -> list[str]:
    # Compatibility with the old shell script, which accidentally passed the
    # literal string "--dataset-parts all" as the option value.
    joined = " ".join(values).replace("--dataset-parts", "").replace("-p", "").strip()
    if joined == "all":
        return ["train", "validation", "test"]
    parts = joined.split()
    if not parts:
        raise ValueError("At least one dataset part is required")
    return parts


def _create_extractor(name: str, codec_checkpoint: Path | None, device: str):
    if name == "encodec":
        return AudioTokenExtractor(AudioTokenConfig(device=device))
    if name == "encodec32":
        return AudioTokenExtractor32(AudioTokenConfig32(device=device))
    if name == "encodec32-ft":
        if codec_checkpoint is None:
            raise ValueError("--codec-checkpoint is required with --audio-extractor encodec32-ft")
        checkpoint = codec_checkpoint.expanduser().resolve()
        return AudioTokenExtractor32FT(
            AudioTokenConfig32(checkpoint=str(checkpoint), device=device)
        )
    return get_fbank_extractor()


def _file_prefix(prefix: str) -> str:
    cleaned = prefix.strip().strip("_")
    return f"{cleaned}_" if cleaned else ""


def tokenize_audio_manifests(args: argparse.Namespace) -> None:
    dataset_parts = _normalize_parts(args.dataset_parts)
    manifest_dir = args.manifest_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=manifest_dir,
        prefix=args.prefix,
        suffix=args.suffix,
        types=["recordings", "supervisions", "cuts"],
    )
    if not manifests:
        raise FileNotFoundError(
            f"No manifests found in {manifest_dir} for parts {dataset_parts}"
        )
    device = str(select_device(args.device))
    logger.info("Audio extractor device: %s", device)
    extractor = _create_extractor(
        args.audio_extractor, args.codec_checkpoint, device
    )
    prefix = _file_prefix(args.prefix)
    storage_name = "fbank" if args.audio_extractor == "fbank" else args.audio_extractor.replace("-", "")

    with get_executor() as executor:
        for partition, manifest in manifests.items():
            logger.info("Processing %s (CUDA=%s)", partition, torch.cuda.is_available())
            if "recordings" in manifest and "supervisions" in manifest:
                cut_set = CutSet.from_manifests(
                    recordings=manifest["recordings"],
                    supervisions=manifest["supervisions"],
                )
            elif "cuts" in manifest:
                cut_set = manifest["cuts"]
            else:
                raise ValueError(f"Incomplete manifest for {partition}: {manifest.keys()}")

            storage_path = output_dir / f"{prefix}{storage_name}_{partition}"
            with torch.no_grad():
                if torch.cuda.is_available() and args.audio_extractor.startswith("encodec"):
                    cut_set = cut_set.compute_and_store_features_batch(
                        extractor=extractor,
                        storage_path=str(storage_path),
                        num_workers=args.num_jobs,
                        batch_duration=args.batch_duration,
                        collate=False,
                        overwrite=args.overwrite,
                        storage_type=NumpyHdf5Writer,
                    )
                else:
                    cut_set = cut_set.compute_and_store_features(
                        extractor=extractor,
                        storage_path=str(storage_path),
                        num_jobs=args.num_jobs,
                        executor=executor,
                        storage_type=NumpyHdf5Writer,
                        overwrite=args.overwrite,
                    )

            for cut in tqdm(cut_set, desc=f"Attach MIDI tokens ({partition})"):
                token_path = Path(cut.supervisions[0].text)
                tokens = np.load(token_path, allow_pickle=False).tolist()
                cut.supervisions[0].custom = {"tokens": {"midi": tokens}}
            cuts_path = output_dir / f"{prefix}cuts_{partition}.{args.suffix}"
            cut_set.to_file(cuts_path)
            logger.info("Wrote %s", cuts_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args()
    tokenize_audio_manifests(args)


if __name__ == "__main__":
    main()
