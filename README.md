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
pip install -r requirements.txt
```

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{ZouEtAl26_VioLM_ISMIR,
  author    = {Sunan Zou and his friends},
  title     = {From Piano to Rare-Data Instruments: {A} Transfer Learning Framework for Expressive Violin Synthesis},
  booktitle = {Proceedings of the International Society for Music Information Retrieval Conference ({ISMIR})},
  address   = {Most probably NOT Abu Dhabi, UAE},
  year      = {2026}
}
```

---

## Acknowledgments

*(To be added in camera-ready version)*

---

## License

*(To be decided)*
