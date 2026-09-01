#!/usr/bin/env python3
# Derived from MIDI-VALLE and modified for VioLM by the VioLM authors, 2026.

"""Prepare aligned MIDI/audio segments for VioLM.

Canonical output layout::

    WORK_DIR/
      midi_segments/{train,validation,test}/*.mid
      audio_segments/{train,validation,test}/*.wav
      metadata/segments.jsonl

The input may either contain split directories or be an unsplit corpus. In the
latter case, a deterministic hash split is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import multiprocessing
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
from tqdm.auto import tqdm

from valle.utils.cli import ConsistentArgumentParser


SPLITS = ("train", "validation", "test")
MIDI_EXTENSIONS = (".mid", ".midi")
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

logger = logging.getLogger("violm.data_prepare")


@dataclass(frozen=True)
class MidiAudioPair:
    midi_path: str
    audio_path: str | None
    item_id: str
    split: str


@dataclass(frozen=True)
class SegmentConfig:
    output_dir: str
    sample_rate: int
    min_seconds: float
    max_seconds: float
    min_tail_seconds: float
    seed: int
    overwrite: bool


def _copy_note(note: pretty_midi.Note, start: float, end: float):
    clipped_start = max(note.start, start) - start
    clipped_end = min(note.end, end) - start
    if clipped_end <= clipped_start:
        return None
    return pretty_midi.Note(
        velocity=note.velocity,
        pitch=note.pitch,
        start=clipped_start,
        end=clipped_end,
    )


def segment_midi(
    midi: pretty_midi.PrettyMIDI, start: float, end: float
) -> pretty_midi.PrettyMIDI:
    """Clip a MIDI object to ``[start, end)`` without duplicating notes."""

    try:
        initial_tempo = float(midi.estimate_tempo())
        if not np.isfinite(initial_tempo) or initial_tempo <= 0:
            initial_tempo = 120.0
    except (ValueError, IndexError, ZeroDivisionError):
        initial_tempo = 120.0
    segment = pretty_midi.PrettyMIDI(
        resolution=midi.resolution,
        initial_tempo=initial_tempo,
    )
    for instrument in midi.instruments:
        output_instrument = pretty_midi.Instrument(
            program=instrument.program,
            is_drum=instrument.is_drum,
            name=instrument.name,
        )
        for note in instrument.notes:
            if note.end > start and note.start < end:
                clipped = _copy_note(note, start, end)
                if clipped is not None:
                    output_instrument.notes.append(clipped)
        for control in instrument.control_changes:
            if start <= control.time < end:
                copied = copy.copy(control)
                copied.time -= start
                output_instrument.control_changes.append(copied)
        for bend in instrument.pitch_bends:
            if start <= bend.time < end:
                copied = copy.copy(bend)
                copied.time -= start
                output_instrument.pitch_bends.append(copied)
        if (
            output_instrument.notes
            or output_instrument.control_changes
            or output_instrument.pitch_bends
        ):
            segment.instruments.append(output_instrument)
    return segment


def _segment_boundaries(
    duration: float,
    min_seconds: float,
    max_seconds: float,
    min_tail_seconds: float,
    seed: int,
) -> list[tuple[float, float]]:
    if duration <= 0:
        return []
    if min_seconds <= 0 or max_seconds < min_seconds:
        raise ValueError("Require 0 < segment-min-seconds <= segment-max-seconds")
    rng = random.Random(seed)
    boundaries = [0.0]
    while duration - boundaries[-1] > max_seconds:
        boundaries.append(
            min(duration, boundaries[-1] + rng.uniform(min_seconds, max_seconds))
        )
    if duration - boundaries[-1] < min_tail_seconds and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(duration)
    return list(zip(boundaries[:-1], boundaries[1:]))


def _stable_split(item: str, ratios: tuple[float, float, float], seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{item}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "validation"
    return "test"


def _split_from_path(relative_path: Path) -> str | None:
    for part in relative_path.parts[:-1]:
        normalized = "validation" if part.lower() in {"valid", "validation", "dev"} else part.lower()
        if normalized in SPLITS:
            return normalized
    return None


def _discover_root(data_dir: Path, explicit: Path | None, candidates: Iterable[str]) -> Path:
    if explicit is not None:
        root = explicit if explicit.is_absolute() else data_dir / explicit
        if not root.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {root}")
        return root
    for name in candidates:
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    return data_dir


def _index_audio(audio_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in audio_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            index.setdefault(path.stem, []).append(path)
    return index


def discover_pairs(
    data_dir: Path,
    midi_dir: Path | None,
    audio_dir: Path | None,
    include_audio: bool,
    split_ratios: tuple[float, float, float],
    seed: int,
    limit: int | None = None,
) -> list[MidiAudioPair]:
    midi_root = _discover_root(data_dir, midi_dir, ("midi", "MIDI"))
    audio_root = None
    audio_index: dict[str, list[Path]] = {}
    if include_audio:
        audio_root = _discover_root(data_dir, audio_dir, ("audio", "wav", "Audio"))
        audio_index = _index_audio(audio_root)

    midi_paths = sorted(
        path
        for path in midi_root.rglob("*")
        if path.is_file() and path.suffix.lower() in MIDI_EXTENSIONS
    )
    if limit is not None:
        midi_paths = midi_paths[:limit]
    if not midi_paths:
        raise FileNotFoundError(f"No .mid/.midi files found under {midi_root}")

    stem_counts: dict[str, int] = {}
    for path in midi_paths:
        stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1

    pairs: list[MidiAudioPair] = []
    missing_audio: list[Path] = []
    for midi_path in midi_paths:
        relative = midi_path.relative_to(midi_root)
        split = _split_from_path(relative) or _stable_split(
            relative.as_posix(), split_ratios, seed
        )
        item_id = midi_path.stem
        if stem_counts[item_id] > 1:
            suffix = hashlib.sha1(relative.as_posix().encode()).hexdigest()[:8]
            item_id = f"{item_id}-{suffix}"

        matched_audio = None
        if include_audio:
            candidates = audio_index.get(midi_path.stem, [])
            if len(candidates) == 1:
                matched_audio = candidates[0]
            elif len(candidates) > 1:
                same_relative = [
                    path
                    for path in candidates
                    if path.relative_to(audio_root).with_suffix("").parts
                    == relative.with_suffix("").parts
                ]
                matched_audio = same_relative[0] if same_relative else sorted(candidates)[0]
                logger.warning(
                    "Multiple audio files match %s; using %s", midi_path, matched_audio
                )
            else:
                missing_audio.append(midi_path)
                continue
        pairs.append(
            MidiAudioPair(
                midi_path=str(midi_path),
                audio_path=str(matched_audio) if matched_audio else None,
                item_id=item_id,
                split=split,
            )
        )

    if missing_audio:
        examples = ", ".join(str(path) for path in missing_audio[:3])
        raise FileNotFoundError(
            f"No stem-matched audio for {len(missing_audio)} MIDI files; examples: {examples}. "
            "Pass --audio-dir explicitly if the corpus uses another layout."
        )
    return pairs


def _pair_seed(pair: MidiAudioPair, base_seed: int) -> int:
    digest = hashlib.sha1(f"{base_seed}:{pair.item_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _process_pair(task: tuple[MidiAudioPair, SegmentConfig]) -> list[dict]:
    pair, config = task
    midi = pretty_midi.PrettyMIDI(pair.midi_path)
    midi_duration = midi.get_end_time()

    waveform = None
    duration = midi_duration
    if pair.audio_path is not None:
        waveform, _ = librosa.load(
            pair.audio_path, sr=config.sample_rate, mono=True
        )
        audio_duration = len(waveform) / config.sample_rate
        duration = min(midi_duration, audio_duration)
        if abs(midi_duration - audio_duration) > 0.5:
            logger.warning(
                "Duration mismatch for %s: MIDI %.3fs, audio %.3fs; clipping to %.3fs",
                pair.item_id,
                midi_duration,
                audio_duration,
                duration,
            )

    records: list[dict] = []
    boundaries = _segment_boundaries(
        duration,
        config.min_seconds,
        config.max_seconds,
        config.min_tail_seconds,
        _pair_seed(pair, config.seed),
    )
    for index, (start, end) in enumerate(boundaries):
        midi_segment = segment_midi(midi, start, end)
        if not any(instrument.notes for instrument in midi_segment.instruments):
            continue
        segment_id = f"{pair.item_id}-{index:04d}"
        midi_output = (
            Path(config.output_dir)
            / "midi_segments"
            / pair.split
            / f"{segment_id}.mid"
        )
        audio_output = (
            Path(config.output_dir)
            / "audio_segments"
            / pair.split
            / f"{segment_id}.wav"
        )
        midi_output.parent.mkdir(parents=True, exist_ok=True)
        if config.overwrite or not midi_output.exists():
            midi_segment.write(str(midi_output))
        if waveform is not None:
            audio_output.parent.mkdir(parents=True, exist_ok=True)
            begin_sample = round(start * config.sample_rate)
            end_sample = round(end * config.sample_rate)
            if config.overwrite or not audio_output.exists():
                sf.write(
                    audio_output,
                    waveform[begin_sample:end_sample],
                    config.sample_rate,
                    subtype="PCM_16",
                )
        records.append(
            {
                "id": segment_id,
                "split": pair.split,
                "source_midi": pair.midi_path,
                "source_audio": pair.audio_path,
                "start": start,
                "duration": end - start,
                "midi": str(midi_output.resolve()),
                "audio": str(audio_output.resolve()) if waveform is not None else None,
            }
        )
    return records


def prepare_segments(args: argparse.Namespace) -> list[dict]:
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    ratios = tuple(args.split_ratios)
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0) or any(value < 0 for value in ratios):
        raise ValueError("--split-ratios must be three non-negative values summing to 1")

    pairs = discover_pairs(
        data_dir=data_dir,
        midi_dir=args.midi_dir,
        audio_dir=args.audio_dir,
        include_audio=args.audio,
        split_ratios=ratios,
        seed=args.seed,
        limit=args.limit,
    )
    config = SegmentConfig(
        output_dir=str(output_dir),
        sample_rate=args.sample_rate,
        min_seconds=args.segment_min_seconds,
        max_seconds=args.segment_max_seconds,
        min_tail_seconds=args.min_tail_seconds,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    tasks = [(pair, config) for pair in pairs]
    all_records: list[dict] = []
    if args.num_workers == 1:
        results = map(_process_pair, tasks)
        for records in tqdm(results, total=len(tasks), desc="Segment MIDI/audio"):
            all_records.extend(records)
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            for records in tqdm(
                pool.imap(_process_pair, tasks),
                total=len(tasks),
                desc="Segment MIDI/audio",
            ):
                all_records.extend(records)

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "segments.jsonl").open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (metadata_dir / "data_prepare_config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "data_dir": str(data_dir),
                "pairs": len(pairs),
                "segments": len(all_records),
                **asdict(config),
            },
            stream,
            indent=2,
        )
    counts = {split: sum(record["split"] == split for record in all_records) for split in SPLITS}
    logger.info("Prepared %d segments: %s", len(all_records), counts)
    return all_records


def get_parser() -> ConsistentArgumentParser:
    parser = ConsistentArgumentParser(
        description="Prepare aligned MIDI/audio segments for VioLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare-segments",
        aliases=["prepare_segments"],
        help="Pair, split, resample, and segment a corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prepare.add_argument(
        "--data-dir", "--data-path", dest="data_dir", type=Path, required=True,
        help="Corpus root. MIDI/audio roots are auto-detected beneath it.",
    )
    prepare.add_argument(
        "--output-dir", "--out-path", dest="output_dir", type=Path, required=True,
        help="Working directory to create.",
    )
    prepare.add_argument("--midi-dir", type=Path, help="MIDI root, absolute or relative to --data-dir.")
    prepare.add_argument("--audio-dir", type=Path, help="Audio root, absolute or relative to --data-dir.")
    prepare.add_argument(
        "--audio", action="store_true",
        help="Require paired audio and create audio_segments. Omit for MIDI-only inference preparation.",
    )
    prepare.add_argument("--sample-rate", type=int, default=32000)
    prepare.add_argument("--segment-min-seconds", type=float, default=15.0)
    prepare.add_argument("--segment-max-seconds", type=float, default=20.0)
    prepare.add_argument("--min-tail-seconds", type=float, default=3.0)
    prepare.add_argument(
        "--split-ratios", type=float, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
        default=(0.8, 0.1, 0.1),
        help="Used only when split directory names are absent.",
    )
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    prepare.add_argument("--limit", type=int, help="Process only the first N MIDI files.")
    prepare.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    parser = get_parser()
    args = parser.parse_args()
    if args.command in {"prepare-segments", "prepare_segments"}:
        prepare_segments(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
