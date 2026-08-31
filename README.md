# VioLM

Official implementation accompanying the ISMIR 2026 manuscript **“VioLM: A
Neural Language Model for Violin Synthesis with Articulation.”** The system
uses a 7-stream articulation-aware MIDI representation and a 32 kHz hybrid
violin/piano neural audio codec.

[Audio demo and qualitative comparisons](https://nii-yamagishilab-visitors.github.io/VioLM/)

## Quick start

Linux and an NVIDIA GPU with CUDA 12.1 are recommended.

```bash
git clone https://github.com/nii-yamagishilab-visitors/VioLM.git
cd VioLM
conda env create -f environment.yml
conda activate violm-ismir26
```

## Environment

Dependency versions and the editable VioLM installation are defined in
`environment.yml`. To update an existing environment after the file changes:

```bash
conda env update -n violm-ismir26 -f environment.yml --prune
```

### Dependency versions

| Component | Version |
|---|---:|
| Python | 3.10.18 |
| PyTorch / torchaudio / torchvision | 2.1.0 / 2.1.0 / 0.16.0 |
| CUDA runtime | 12.1 |
| AudioCraft | 1.3.0 |
| FFmpeg / PyAV | 6.1 / 11.0.0 |
| Lhotse | 1.32.2 |
| MidiTok / Symusic | 3.0.6.post1 / 0.6.0 |
| librosa / soundfile | 0.11.0 / 0.12.1 |
| torchmetrics | 0.11.1 |
| transformers | 4.41.2 |
| loralib | 0.1.2 |
| setuptools | 75.3.0 |

`setuptools` 75.3.0 provides `pkg_resources`, which is required by the pinned
PrettyMIDI and torchmetrics versions. PyAV 11 is paired with FFmpeg 6.

### GPU selection

Audio tokenization, single-GPU training, and inference accept `--device`.
`--device auto` selects the visible CUDA GPU with the most free memory. An
explicit device such as `cuda:2` or `cpu` can also be used. Multi-GPU DDP maps
ranks to visible CUDA devices; use `CUDA_VISIBLE_DEVICES` to control that
mapping.

## Working directory layout

The data-preparation commands create the following files under `WORK_DIR`:

```text
WORK_DIR/
├── midi_segments/{train,validation,test}/*.mid
├── audio_segments/{train,validation,test}/*.wav
├── midi_tokens/{train,validation,test}/*.npy
├── manifests/{recordings,supervisions}_{split}.jsonl.gz
├── audio_tokens/
│   ├── encodec32ft_{split}.h5
│   └── cuts_{split}.jsonl.gz
└── metadata/{segments.jsonl,data_prepare_config.json}
```

The `.npy` files contain `[notes, 7]` arrays with one BOS row and one EOS row.
The final audio features contain four codec indices per frame at 50 Hz.

## Data preparation

The open-source dataset is available from
[Zenodo](https://zenodo.org/records/21670078).

Store the dataset outside the repository using the following format:

```text
DATA_DIR/
├── midi/**/*.mid
└── audio/**/*.{wav,flac,mp3,ogg,m4a}
```

`.midi` files are also supported. Each MIDI/audio pair must share a filename
stem. The `midi` and `audio` trees may contain `train`, `validation`, and
`test` directories; these splits are preserved. When split directories are
absent, the files are assigned a deterministic 8:1:1 split using `--seed`.

```bash
DATA_DIR=/path/to/dataset
WORK_DIR=/path/to/violm-work
CODEC_CHECKPOINT=/path/to/compression_32khz_violin_piano_hybrid.bin

python transfer/data_prepare.py prepare-segments \
  --data-dir "$DATA_DIR" \
  --midi-dir midi \
  --audio-dir audio \
  --output-dir "$WORK_DIR" \
  --audio \
  --sample-rate 32000 \
  --segment-min-seconds 15 \
  --segment-max-seconds 20 \
  --min-tail-seconds 3 \
  --seed 42

python transfer/midi_tokenize.py \
  --input-dir "$WORK_DIR/midi_segments" \
  --output-dir "$WORK_DIR/midi_tokens"

python transfer/cut_creation.py \
  --audio-dir "$WORK_DIR/audio_segments" \
  --midi-token-dir "$WORK_DIR/midi_tokens" \
  --output-dir "$WORK_DIR/manifests"

bash transfer/audio_tokenization.sh \
  --work-dir "$WORK_DIR" \
  --codec-checkpoint "$CODEC_CHECKPOINT" \
  --device auto
```

Add `--overwrite` when generated outputs should be replaced.

## Training

The paper architecture is 12 layers, 16 heads, width 1024, and four audio
quantizers. This example performs rank-8 LoRA adaptation. The initialization
checkpoint must use the same architecture and MIDI stream count selected on
the command line.

```bash
BASE_CHECKPOINT=/path/to/compatible-base-checkpoint.pt
EXP_DIR=/path/to/experiments/violm-lora-r8

python model/trainer.py \
  --device auto \
  --manifest-dir "$WORK_DIR/audio_tokens" \
  --exp-dir "$EXP_DIR" \
  --pretrained-checkpoint "$BASE_CHECKPOINT" \
  --model-name valle \
  --decoder-dim 1024 \
  --nhead 16 \
  --num-decoder-layers 12 \
  --num-quantizers 4 \
  --prefix-mode 1 \
  --articulation true \
  --norm-first true \
  --add-prenet false \
  --share-embedding true \
  --use-lora true \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-target-modules all \
  --lora-freeze-base true \
  --num-epochs 50 \
  --max-duration 200 \
  --num-buckets 12 \
  --base-lr 0.05 \
  --warmup-steps 20 \
  --accumulate-grad-steps 8 \
  --save-every-n 5000 \
  --valid-interval 2000 \
  --average-period 0
```

To save decoded audio during training, add
`--save-training-audio true --codec-checkpoint "$CODEC_CHECKPOINT"`. When
training `bert-valle`, also pass `--bert-checkpoint /path/to/checkpoint`.

## Inference

Inference consumes tokenized MIDI `.npy` files rather than raw MIDI. Prompt
MIDI/audio pairs must be aligned. LoRA usage, rank, targets, model width,
layer count, quantizer count, and articulation support are inferred from the
checkpoint, so they do not need to be repeated.

```bash
MODEL_CHECKPOINT=/path/to/best-valid-loss.pt
PROMPT_TOKENS=/path/to/prompt.npy
PROMPT_AUDIO=/path/to/prompt.wav
TARGET_TOKENS=/path/to/target.npy
OUTPUT_DIR=/path/to/generated

python model/infer.py \
  --checkpoint "$MODEL_CHECKPOINT" \
  --codec-checkpoint "$CODEC_CHECKPOINT" \
  --midi-prompts "$PROMPT_TOKENS" \
  --audio-prompts "$PROMPT_AUDIO" \
  --midi-tokens "$TARGET_TOKENS" \
  --output-dir "$OUTPUT_DIR" \
  --output-file violm.wav \
  --device auto \
  --seed 42
```

Multiple prompt pairs can be listed after `--midi-prompts` and
`--audio-prompts`; internal BOS/EOS rows are removed automatically. Existing
output files are skipped unless `--overwrite` is supplied.

## Synthetic-data rendering

`data/synthesis_vst.py` renders MIDI files with an external REAPER instance.
Install `python-reapy`, enable the REAPER ReaScript API, and configure the VST
instrument before running the module:

```bash
pip install python-reapy
python data/synthesis_vst.py \
  --input-dir /path/to/midi \
  --output-dir /path/to/audio \
  --plugin "VST3: Violin Plugin Name"
```

## Citation

```bibtex
@inproceedings{ZouEtAl26VioLM,
  author    = {Sunan Zou and Zhe Zhang and Yigitcan {\"O}zer and Guojie Luo and Junichi Yamagishi},
  title     = {{VioLM}: A Neural Language Model for Violin Synthesis with Articulation},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference},
  year      = {2026}
}
```

## Acknowledgments

VioLM builds on [MIDI-VALLE](https://github.com/nii-yamagishilab/MIDI-VALLE),
AudioCraft/EnCodec, MidiTok, Symusic, and Lhotse.
