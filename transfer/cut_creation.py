#!/usr/bin/env python3
"""Create Lhotse recording/supervision manifests for VioLM."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.supervision import SupervisionSegment, SupervisionSet
from tqdm.auto import tqdm

from valle.utils.cli import ConsistentArgumentParser


DEFAULT_SPLITS = ("train", "validation", "test")
logger = logging.getLogger("violm.cut_creation")


def _metadata_from_id(item_id: str) -> tuple[str, str]:
    parts = item_id.split("_")
    speaker = parts[0] if parts else "unknown"
    language_or_style = parts[1] if len(parts) >= 2 else "0"
    return speaker, language_or_style


def prepare_manifests(
    audio_dir: Path,
    midi_token_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    allow_missing: bool = False,
) -> dict[str, dict[str, RecordingSet | SupervisionSet]]:
    audio_dir = audio_dir.resolve()
    midi_token_dir = midi_token_dir.resolve()
    output_dir = output_dir.resolve()
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio segment directory does not exist: {audio_dir}")
    if not midi_token_dir.is_dir():
        raise FileNotFoundError(f"MIDI token directory does not exist: {midi_token_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, RecordingSet | SupervisionSet]] = defaultdict(dict)
    missing_tokens: list[Path] = []
    for split in tqdm(splits, desc="Create manifests"):
        recordings = []
        supervisions = []
        split_audio_dir = audio_dir / split
        audio_paths = sorted(split_audio_dir.rglob("*.wav")) if split_audio_dir.is_dir() else []
        for audio_path in audio_paths:
            relative = audio_path.relative_to(split_audio_dir).with_suffix(".npy")
            token_path = midi_token_dir / split / relative
            if not token_path.is_file():
                missing_tokens.append(token_path)
                continue
            recording = Recording.from_file(audio_path)
            if recording.duration <= 0:
                logger.warning("Skipping zero-duration audio: %s", audio_path)
                continue
            item_id = audio_path.stem
            speaker, language = _metadata_from_id(item_id)
            recordings.append(recording)
            supervisions.append(
                SupervisionSegment(
                    id=item_id,
                    recording_id=recording.id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    speaker=speaker,
                    language=language,
                    text=str(token_path.resolve()),
                )
            )

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        recording_set, supervision_set = fix_manifests(
            recording_set, supervision_set
        )
        validate_recordings_and_supervisions(recording_set, supervision_set)
        recording_set.to_file(output_dir / f"recordings_{split}.jsonl.gz")
        supervision_set.to_file(output_dir / f"supervisions_{split}.jsonl.gz")
        manifests[split] = {
            "recordings": recording_set,
            "supervisions": supervision_set,
        }
        logger.info("%s: %d aligned items", split, len(recording_set))

    if missing_tokens and not allow_missing:
        examples = ", ".join(str(path) for path in missing_tokens[:3])
        raise FileNotFoundError(
            f"Missing {len(missing_tokens)} MIDI token files; examples: {examples}"
        )
    if missing_tokens:
        logger.warning("Skipped %d audio files without MIDI tokens", len(missing_tokens))
    return manifests


def _first_existing(candidates: list[Path]) -> Optional[Path]:
    return next((path for path in candidates if path.is_dir()), None)


def _resolve_legacy_layout(args, parser) -> tuple[Path, Path]:
    audio_dir = args.audio_dir
    midi_token_dir = args.midi_token_dir
    if args.corpus_dir is not None:
        corpus_dir = args.corpus_dir
        audio_dir = audio_dir or _first_existing(
            [corpus_dir / "audio_segments", corpus_dir / "wav_seg"]
        )
        midi_token_dir = midi_token_dir or _first_existing(
            [corpus_dir / "midi_tokens", corpus_dir / "midi_seg"]
        )
    if audio_dir is None:
        parser.error("--audio-dir is required (or pass --corpus-dir for legacy auto-detection)")
    if midi_token_dir is None:
        parser.error("--midi-token-dir is required (or pass --corpus-dir for legacy auto-detection)")
    return audio_dir, midi_token_dir


def get_parser() -> ConsistentArgumentParser:
    parser = ConsistentArgumentParser(
        description="Create Lhotse manifests from VioLM segments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio-dir", type=Path, help="Root of audio_segments.")
    parser.add_argument("--midi-token-dir", type=Path, help="Root of midi_tokens.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Manifest output directory.")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    parser = get_parser()
    args = parser.parse_args()
    audio_dir, midi_token_dir = _resolve_legacy_layout(args, parser)
    prepare_manifests(
        audio_dir=audio_dir,
        midi_token_dir=midi_token_dir,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
