# VioLM: From Piano to Rare-Data Instruments

**A Transfer Learning Framework for Expressive Violin Synthesis**

> To be submitted to ISMIR 2026

---

## Overview

VioLM addresses the challenge of expressive music performance synthesis (MPS) for instruments with scarce aligned data. Taking violin as a case study, the framework combines a synthetic data generation paradigm with a two-stage transfer learning scheme built on top of a pre-trained piano model (MIDI-VALLE).

Key contributions:

- An expressive aligned data generation paradigm for rare-data instruments, using GigaMIDI as MIDI candidates and a DAW-based synthesis pipeline
- A two-stage transfer learning scheme adapting VALL-E from piano to violin
- Explicit articulation modeling (legato, detache, pizzicato, harmonics, staccato) in NN-based synthesis for the first time
- Open-sourced violin articulation dataset

---

## Method

### 1. Synthetic Data Preparation

MIDI files from [GigaMIDI](https://github.com/Metacreation-Lab/GigaMIDI-Dataset) are filtered and preprocessed for violin playability (pitch range, polyphony constraints), automatically annotated with articulations and velocities, and synthesized using multiple VSTs in a DAW. Loudness calibration is applied per articulation across VSTs.

### 2. Tokenization

An extended Octuple MIDI tokenization scheme is used, augmenting standard note features (pitch, velocity, duration, inter-onset interval, position, bar index) with one-hot articulation and vibrato labels.

### 3. Transfer Learning

- **Stage 1 (middle training):** Finetune the MIDI-VALLE language model on synthetic violin data; finetune EnCodec on a balanced mix of synthetic and real data.
- **Stage 2 (post-training):** Finetune EnCodec solely on real violin performances for high-fidelity audio generation.

---

## Repository Structure

```
VioLM/
├── data/               # Data preparation and preprocessing scripts
├── tokenization/       # Octuple + articulation tokenization
├── model/              # Model architecture and training code
├── transfer/           # Two-stage transfer learning scripts
├── eval/               # Objective and subjective evaluation
└── configs/            # Experiment configuration files
```

*(To be populated)*

---

## Installation

```bash
git clone https://github.com/nii-yamagishilab-visitors/VioLM.git
cd VioLM
conda create -n violm python=3.10.18
conda activate violm

# Download pkg-config,ffmpeg, and gcc with conda (no sudo needed)
conda install -c conda-forge pkg-config ffmpeg=5 gcc gxx_linux-64



# AudioCraft
git clone https://github.com/facebookresearch/audiocraft.git; cd audiocraft; pip install -e .; cd ..
# PyTorch
pip install torch==2.1.0 torchaudio==2.1.0
pip install torchmetrics==0.11.1

#h5py
pip install h5py

# fbank
pip install librosa==0.11.0

# lhotse
pip uninstall lhotse
pip install git+https://github.com/lhotse-speech/lhotse

# k2
# find the right version in https://huggingface.co/csukuangfj/k2
pip install https://huggingface.co/csukuangfj/k2/resolve/main/cpu/k2-1.24.4.dev20240223+cpu.torch2.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# icefall
git clone https://github.com/k2-fsa/icefall
cd icefall
pip install -r requirements.txt
export PYTHONPATH=`pwd`/../icefall:$PYTHONPATH
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.zshrc
echo "export PYTHONPATH=`pwd`/../icefall:\$PYTHONPATH" >> ~/.bashrc
cd -
source ~/.zshrc

# VioLM (editable install)
pip install -e .
```

---

## Data processing (for training)

This section follows the same logic as `MIDI-VALLE/egs/atepp`, but adapted to this repo.

### 1) Segment audio + MIDI

```bash
# Example: create segmented MIDI/WAV under /
python3 transfer/data_prepare.py prepare_segments \
  --data_path "/path/to/your/dataset" \
  --out_path "/path/to/your/working/dataset" \
  --audio
```

This creates (at least):

- `/midi_seg/{train,validation,test}/*.midi`
- `/wav_seg/{train,validation,test}/*.wav`

### 2) Tokenize the segmented MIDI into `.npy`

```bash
# Tokenize all splits under `midi_seg/`
python3 transfer/midi_tokenize.py \
  --data_folder "/path/to/your/working/dataset" \
  --target_folder "midi_seg"
```

### 3) Create Lhotse manifests (recordings + supervisions)

```bash
# Creates e.g. /manifests_articulation/{recordings_*.jsonl.gz, supervisions_*.jsonl.gz}
python3 transfer/cut_creation.py \
  --corpus-dir "/path/to/your/working/dataset" \
  --output-dir "/path/to/your/working/manifests"
```

### 4) Tokenize audio into EnCodec feature files + cuts manifests

You renamed `prepare.sh` to `transfer/audio_tokenization.sh`. This step should produce:

- `cuts_train.jsonl.gz`, `cuts_validation.jsonl.gz`, `cuts_test.jsonl.gz` inside your `--manifest-dir`
- the EnCodec HDF5 files referenced by those cuts (e.g. `*_encodec_train.h5`)

This repo’s `audio_tokenization.sh` runs the tokenizer with `--prefix ""`, so you should see filenames like:

- `cuts_*.jsonl.gz` (no `atepp_` prefix)
- `_encodec_*.h5` (leading underscore because the prefix is empty)

After this, your trainer can be pointed to the directory containing `cuts_*.jsonl.gz` via:

- `--manifest-dir "/tokenized_articulation"` (or your chosen output directory)

```bash
# Run from VioLM/
bash transfer/audio_tokenization.sh
```

---
## Training

Once you have dataset prepared, you can train with:

```bash
conda activate violm

python3 model/trainer.py \
  --num-buckets 12 \
  --save-every-n 5000 --valid-interval 2000 \
  --model-name valle --share-embedding true --norm-first true --add-prenet false \
  --decoder-dim 1024 --nhead 16 --num-decoder-layers 12 --prefix-mode 1 \
  --base-lr 0.05 --warmup-steps 20 --average-period 0 \
  --num-epochs 50 --start-epoch 1 --start-batch 0 --accumulate-grad-steps 8 \
  --max-duration 200 --keep-last-k 3 \
  --train-stage 0 --num-quantizers 4 \
  --filter-min-duration 3 \
  --inf-check True \
  --world-size 1 \                                      # Number of GPUs (data parallelism)
  --articulation True \                                 # Use articulation-augmented dataset
  --manifest-dir "local/MUSC/tokenized_articulation" \  # Tokenized dataset manifests
  --exp-dir exp/violin-model-arti-MUSC-lora_r8          # Experiment output directory
```

LoRA and full-size finetuning are also supported if you add the following parameters:
```bash
--use-lora true --lora-rank 8 --lora-alpha 16 --lora-target-modules all \
```
```bash
  --pretrained-checkpoint "path/to/pretrained/model"
```

Notes:
- `--num-buckets` must match the configuration used to train your `--pretrained-checkpoint` (otherwise you may get embedding size mismatches).

---
## Inference

`model/infer.py` expects tokenized MIDI tokens (`.npy`), not raw `.midi`.
Those `.npy` token files are produced by `transfer/midi_tokenize.py`.

Example:

```bash
conda activate violm

python3 model/infer.py \
  --checkpoint "checkpoint/best-valid-loss.pt" \
  --midi-prompts "<PROMPT>.npy" \
  --audio-prompts "<PROMPT>.wav" \
  --midi "<TARGET>.npy" \
  --output-dir "path/to/output/directory" \
  --output-file <TARGET>.wav \
  --use-lora true --lora-rank 8 --lora-alpha 16 --lora-target-modules all (must match with your training settings)
```

---

## Citation

If you use this work, please cite:

```bibtex
<!-- @inproceedings{ZouEtAl26_VioLM_ISMIR,
  author    = {Sunan Zou and Zhe Zhang and Yigitcan Özer and Guojie Lou and Junichi Yamagishi},
  title     = {VioLM: A Neural Language Model for Violin Synthesis with Articulation},
  booktitle = {Proceedings of the 27th International Society for Music Information Retrieval Conference ({ISMIR})},
  address   = {Abu Dhabi, UAE},
  year      = {2026}
} -->
```

---

## Acknowledgments

*(To be added in camera-ready version)*

---

## License

*(To be decided)*
