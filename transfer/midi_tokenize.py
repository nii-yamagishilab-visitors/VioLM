import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from miditok import TokenizerConfig


BOS_ID = 1
EOS_ID = 2


def _load_expression_tok(repo_root: Path):
    """
    Load ExpressionTok from `tokenization/VioLM-tokenizer.py`.

    The file name contains a hyphen, so we load it by path instead of normal import.
    """
    import importlib.util

    tok_path = repo_root / "tokenization" / "VioLM-tokenizer.py"
    spec = importlib.util.spec_from_file_location("violm_tokenizer", tok_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load tokenizer module from {tok_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.ExpressionTok


def _load_tokenizer_config(repo_root: Path) -> TokenizerConfig:
    cfg_path = repo_root / "tokenization" / "VioLM-tokenizer.json"
    data: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    # The JSON is saved as {"config": {...}, ...}
    cfg = data["config"]
    # The saved JSON may contain non-int keys (e.g. time_signature_range)
    # that need normalization before feeding into TokenizerConfig.
    def _range_key_to_tuple(k: str):
        a, b = k.split("_")
        return (int(a), int(b))

    if isinstance(cfg.get("beat_res"), dict):
        cfg["beat_res"] = {
            _range_key_to_tuple(k): v for k, v in cfg["beat_res"].items()
        }
    if isinstance(cfg.get("beat_res_rest"), dict):
        cfg["beat_res_rest"] = {
            _range_key_to_tuple(k): v for k, v in cfg["beat_res_rest"].items()
        }
    if isinstance(cfg.get("time_signature_range"), dict):
        cfg["time_signature_range"] = {
            int(k): v for k, v in cfg["time_signature_range"].items()
        }
    config = TokenizerConfig(**cfg)
    # Ensure required fields exist for ExpressionTok
    # (miditok may drop/override missing additional_params)
    ap = dict(getattr(config, "additional_params", {}) or {})
    ap.setdefault("data_type", "Midi")
    config.additional_params = ap
    return config

def process_with_expression_tokenizer(data_folder: str, target_folder: str):
    # Assume this script lives in `VioLM/transfer/`
    repo_root = Path(__file__).resolve().parents[1]

    ExpressionTok = _load_expression_tok(repo_root)
    tokenizer = ExpressionTok(_load_tokenizer_config(repo_root))

    corpus_dir = Path(data_folder)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    transcript_dir = corpus_dir / target_folder
    transcript_files = transcript_dir.rglob("*.midi")
    slens = {}
    
    for file in tqdm(transcript_files):
        # if os.path.isfile(file.parent / (str(file.stem) + ".npy")):
        #     continue
        
        try:
            sequence = tokenizer(file)[0].ids
        except Exception:
            os.remove(file)
            os.remove(str(file).replace("midi_seg", "wav_seg").replace("midi", "wav"))
            print(f"Error dealing with: {file.stem}")
            # print(file)
            continue 
        
        # If the sequence is empty, remove the file
        if len(sequence) == 0:
            try:
                os.remove(file)
                os.remove(str(file).replace("midi_seg", "wav_seg").replace("midi", "wav"))
                print(f"File Removed: {file.stem}")
                continue
            except:
                print(f"File Not Found: {file.stem}")
                continue
        #print(sequence[0])
        category_length = len(sequence[0])
        
        # Add BOS and EOS tokens
        bos_row = [[BOS_ID] * category_length]
        eos_row = [[EOS_ID] * category_length]
        sequence = bos_row + sequence + eos_row
        slens[str(file.stem)] = len(sequence)
        
        np.save(file.parent / (str(file.stem) + ".npy"), sequence)
        
    # np.savez("valle-stats.npz", slens=slens)
    slens_values = [v for v in slens.values()]
    print(f"The minimum sequenc lens: {min(slens_values)}")
    print(f"The maximum sequenc lens: {max(slens_values)}")
    print(f"average lens: {sum(slens_values)/len(slens_values)}")
    print((np.asarray(slens_values)< 100).sum())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MIDI files with ExpressionTok")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder containing MIDI files")
    parser.add_argument("--target_folder", type=str, default="midi_seg", help="Target folder to save processed files")
    
    args = parser.parse_args()
    
    process_with_expression_tokenizer(args.data_folder, args.target_folder)
    print("Processing completed.")