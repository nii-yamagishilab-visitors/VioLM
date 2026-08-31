from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import torch

from model.infer import _concat_token_files, _lora_targets_from_state
from transfer.data_prepare import _segment_boundaries, segment_midi
from valle.utils.checkpoint import load_checkpoint, save_checkpoint
from valle.utils.cli import ConsistentArgumentParser
from valle.utils.runtime import AttributeDict, make_pad_mask


def test_legacy_cli_underscore_is_normalized() -> None:
    parser = ConsistentArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    with pytest.warns(FutureWarning, match="--output_dir is deprecated"):
        args = parser.parse_args(["--output_dir", "result"])
    assert args.output_dir == Path("result")


def test_segment_boundaries_are_deterministic_and_complete() -> None:
    first = _segment_boundaries(9.5, 2.0, 3.0, 0.5, seed=42)
    second = _segment_boundaries(9.5, 2.0, 3.0, 0.5, seed=42)
    assert first == second
    assert first[0][0] == 0.0
    assert first[-1][1] == pytest.approx(9.5)
    assert all(end > start for start, end in first)


def test_segment_midi_clips_crossing_note_once() -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=40)
    instrument.notes.append(
        pretty_midi.Note(velocity=90, pitch=69, start=1.0, end=3.0)
    )
    midi.instruments.append(instrument)

    clipped = segment_midi(midi, 2.0, 4.0)
    notes = clipped.instruments[0].notes
    assert len(notes) == 1
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(1.0)


def test_prompt_token_concatenation_removes_internal_boundaries(tmp_path) -> None:
    first = np.array([[1] * 7, [10] * 7, [2] * 7], dtype=np.int64)
    second = np.array([[1] * 7, [20] * 7, [2] * 7], dtype=np.int64)
    paths = [tmp_path / "first.npy", tmp_path / "second.npy"]
    np.save(paths[0], first)
    np.save(paths[1], second)

    combined = _concat_token_files(paths)
    assert combined.tolist() == [
        [1] * 7,
        [10] * 7,
        [20] * 7,
        [2] * 7,
    ]


def test_lora_targets_detect_separate_qkv() -> None:
    state = {
        "layer.linear1.lora_A": torch.zeros(2, 2),
        "layer.self_attn.out_proj.lora_A": torch.zeros(2, 2),
        "layer.self_attn.q_proj.lora_A": torch.zeros(2, 2),
        "layer.self_attn.k_proj.lora_A": torch.zeros(2, 2),
        "layer.self_attn.v_proj.lora_A": torch.zeros(2, 2),
    }
    targets = set(_lora_targets_from_state(state).split("+"))
    assert targets == {"ffn", "attn_out", "attn_q", "attn_k", "attn_v"}


def test_checkpoint_round_trip(tmp_path) -> None:
    source = torch.nn.Linear(2, 3)
    path = tmp_path / "checkpoint.pt"
    params = AttributeDict(epoch=1, batch_idx_train=3)
    save_checkpoint(path, source, params)

    restored = torch.nn.Linear(2, 3)
    metadata = load_checkpoint(path, restored)
    assert metadata["epoch"] == 1
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_make_pad_mask() -> None:
    mask = make_pad_mask(torch.tensor([2, 1]), max_len=3)
    assert mask.tolist() == [[False, False, True], [False, True, True]]
