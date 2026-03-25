#!/usr/bin/env python3
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Phonemize midi and EnCodec Audio.

Usage example:
    python3 bin/infer.py \
        --decoder-dim 128 --nhead 4 --num-decoder-layers 4 --model-name valle \
        --midi-prompts "Go to her." \
        --audio-prompts ./prompts/61_70970_000007_000001.wav \
        --output-dir infer/demo_valle_epoch20 \
        --checkpoint exp/valle_nano_v2/epoch-20.pt

"""
import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import glob
import random

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio
from icefall.utils import AttributeDict, str2bool

from valle.data import (
    AudioTokenizer,
    AudioTokenizer32,
    AudioTokenizer32FT,
    tokenize_audio,
)
from valle.data.collation import get_midi_token_collater, get_midi_token_collater
from valle.models import add_model_arguments, get_model
from valle.models.lora_valle import (
    apply_lora_to_valle,
    _parse_targets,
    use_attn_qkv_linear,
    use_attn_separate_qkv_linear,
)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--midi-prompts",
        type=str,
        default="",
        help="midi prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --midi-prompts.",
    )

    parser.add_argument(
        "--midi",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="midi to be synthesized.",
    )

    parser.add_argument(
        "--midi-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--continual",
        type=str2bool,
        default=False,
        help="Do continual task.",
    )
    
    parser.add_argument(
        "--demo",
        type=str2bool,
        default=False,
        help="Do demo task.",
    )
    
    parser.add_argument(
        "--audio-tokenizer",
        type=str,
        default="Encodec",
        help="Which audio tokenizer to use, allow (Encodec, Encodec32, Encodec32FT)",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="valle",
        help="Output file name",
    )

    # LoRA-related arguments (should match training settings for LoRA checkpoints)
    parser.add_argument(
        "--use-lora",
        type=str2bool,
        default=False,
        help="Expect a LoRA-augmented VALLE checkpoint and construct the model with LoRA adapters.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank used during training.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha used during training.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout used during training.",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="ffn+attn_out",
        help=(
            "LoRA targets used during training. Individual targets: 'ffn', 'attn_out', "
            "'attn_q', 'attn_k', 'attn_v'. Shorthands: 'attn' -> attn_out+attn_q+attn_k+attn_v, "
            "'all' -> ffn+attn_out+attn_q+attn_k+attn_v, 'attn_qkv' -> legacy fused Q/K/V. "
            "Combine with '+', e.g. 'ffn+attn' or 'ffn+attn_out+attn_q'."
        ),
    )

    add_model_arguments(parser)

    return parser.parse_args()


def load_model(
    checkpoint,
    device,
    tokenizer_type: str = "articulation",
    use_lora: bool = False,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    lora_target_modules: str = "ffn+attn_out",
):
    if not checkpoint:
        return None

    checkpoint = torch.load(checkpoint, map_location=device)

    # Auto-detect articulation from embedding input dimension (560 = 7 streams, 544 = 6 streams)
    ckpt_emb_in = checkpoint["model"]["ar_text_embedding.weight"].shape[1]
    articulation = (ckpt_emb_in == 560)

    args = AttributeDict(checkpoint)
    args["model_name"] = "valle"
    args["decoder_dim"] = 1024
    args["nhead"] = 16
    args["num_decoder_layers"] = 12
    args["norm_first"] = True
    args["add_prenet"] = False
    args["prefix_mode"] = 1
    args["share_embedding"] = True
    args["scale_factor"] = 1.0
    args["prepend_bos"] = False
    args["num_quantizers"] = 4
    args["tokenizer_type"] = tokenizer_type
    args["articulation"] = articulation

    from contextlib import nullcontext
    if use_lora:
        _targets = _parse_targets(lora_target_modules)
        _use_separate_qkv = bool(_targets & {"attn_q", "attn_k", "attn_v"})
        _use_fused_qkv = "attn_qkv" in _targets
    else:
        _use_separate_qkv = _use_fused_qkv = False

    if _use_separate_qkv:
        build_ctx = use_attn_separate_qkv_linear()
    elif _use_fused_qkv:
        build_ctx = use_attn_qkv_linear()
    else:
        build_ctx = nullcontext()
    with build_ctx:
        model = get_model(args)

    # If the checkpoint was trained with LoRA, we must construct the same
    # LoRA-augmented architecture before loading the state dict.
    if use_lora:
        logging.info(
            f"Constructing LoRA-augmented VALLE for inference "
            f"(rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
            f"targets='{lora_target_modules}')"
        )
        apply_lora_to_valle(
            model,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_target_modules,
            freeze_base=False,
        )

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint["model"], strict=True
    )
    assert not missing_keys
    model.to(device)
    model.eval()

    return model

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    model = load_model(
        args.checkpoint,
        device,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    midi_collater = get_midi_token_collater()

    audio_tokenizer = AudioTokenizer32FT()
        
    if os.path.isfile(f"{args.output_dir}/{args.output_file}.wav"):
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    midi_prompts = " ".join(args.midi_prompts.split("|"))
    audio_prompts = []

    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_audio(args.audio_tokenizer, audio_file, audio_tokenizer)
            audio_prompts.append(encoded_frames)

        assert len(args.midi_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)
    # print(midi_prompts)
    midi_prompts = np.load(midi_prompts)

    model.eval()

    for n, midi in enumerate(args.midi.split("|")):
        if midi != "":
            midi = np.load(midi, allow_pickle=True)

            midi_tokens, midi_tokens_lens = midi_collater(
                [   
                    #(NOTE:Jingjing) Ensure the extra BOS & EOS was removed
                    # np.concatenate([midi_prompts[:-1], midi[1:]], axis=0).tolist() 
                    midi.tolist()
                ]
            )
        else:
            midi_tokens, midi_tokens_lens = midi_collater(
                [
                    midi_prompts.tolist()
                ]
            )

        with torch.no_grad():
            # synthesis
            if args.continual:
                assert midi == ""
                encoded_frames = model.continual(
                    midi_tokens.to(device),
                    midi_tokens_lens.to(device),
                    audio_prompts,
                )
            else:
                enroll_x_lens = None
                if len(midi_prompts) != 0:
                    _, enroll_x_lens = midi_collater(
                        [
                            midi_prompts[:-1].tolist()
                        ]
                    )
                encoded_frames = model.inference(
                    midi_tokens.to(device),
                    midi_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

            if audio_prompts != []:
                samples = audio_tokenizer.decode(
                    encoded_frames.transpose(2, 1), None
                )
                # store
                torchaudio.save(
                    f"{args.output_dir}/{args.output_file}_{n}.wav", samples[0].cpu(), 32000
                )
            else:  # Transformer
                pass

if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
