#!/usr/bin/env python3
"""Apply a fixed decibel gain to WAV files."""

import os

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from valle.utils.cli import ConsistentArgumentParser


def db_to_amplitude(db):
    """Convert decibels to amplitude gain factor."""
    return librosa.db_to_amplitude(db, ref=1.0)

def amplitude_to_db(amplitude):
    """Convert amplitude to decibels."""
    return librosa.amplitude_to_db(amplitude, ref=1.0)

def amplify_by_db(input_dir, output_dir, db_increase):
    """
    Amplify all .wav files by a specified dB amount.
    
    Parameters:
    - input_dir: directory containing input .wav files
    - output_dir: directory to save processed files
    - db_increase: amount in dB to increase amplitude (e.g., 2.27)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate gain factor from dB
    gain = db_to_amplitude(db_increase)
    
    # Get all .wav files in the input directory
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} .wav files to process")
    print(f"dB increase: {db_increase:.2f} dB")
    print(f"Gain factor: {gain:.4f}x\n")
    
    for filename in wav_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load audio file
            audio, sr = librosa.load(input_path, sr=None, mono=False)
            
            # Get original max amplitude
            orig_max = np.max(np.abs(audio))
            orig_max_db = amplitude_to_db(orig_max)
            
            # Apply gain
            amplified_audio = audio * gain
            
            # Get new max amplitude
            new_max = np.max(np.abs(amplified_audio))
            new_max_db = amplitude_to_db(new_max)
            
            # Save the amplified audio
            sf.write(output_path, amplified_audio.T if audio.ndim > 1 else amplified_audio, sr)
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}\n")
    
    print("Processing complete!")

if __name__ == "__main__":
    parser = ConsistentArgumentParser(description="Apply a fixed dB gain to WAV files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--db-increase", type=float, required=True)
    args = parser.parse_args()
    amplify_by_db(args.input_dir, args.output_dir, args.db_increase)
