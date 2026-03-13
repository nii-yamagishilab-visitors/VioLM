import os
import wave
import numpy as np
from pathlib import Path

def has_leading_silence(wav_path, silence_threshold_seconds=2.0, silence_db=-40):
    """
    Check if a WAV file starts with silence >= threshold.
    
    Args:
        wav_path: Path to WAV file
        silence_threshold_seconds: Minimum silence duration to check (default 2.0s)
        silence_db: dB threshold for silence (default -40dB)
    
    Returns:
        True if file starts with >= threshold seconds of silence
    """
    try:
        with wave.open(str(wav_path), 'rb') as wav:
            # Get audio properties
            sample_rate = wav.getframerate()
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            
            # Calculate frames to check
            frames_to_check = int(sample_rate * silence_threshold_seconds)
            
            # Read audio data
            audio_data = wav.readframes(frames_to_check)
            
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8
                audio = np.frombuffer(audio_data, dtype=dtype)
                audio = (audio.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                dtype = np.int16
                audio = np.frombuffer(audio_data, dtype=dtype)
                audio = audio.astype(np.float32) / 32768.0
            else:
                dtype = np.int32
                audio = np.frombuffer(audio_data, dtype=dtype)
                audio = audio.astype(np.float32) / 2147483648.0
            
            # Reshape for multi-channel audio
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
                # Use max across channels
                audio = np.max(np.abs(audio), axis=1)
            else:
                audio = np.abs(audio)
            
            # Convert to dB
            audio_db = 20 * np.log10(audio + 1e-10)  # Add small value to avoid log(0)
            
            # Check if all samples are below silence threshold
            return np.all(audio_db < silence_db)
            
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return False

def remove_silent_wav_files(directory, silence_threshold=2.0, silence_db=-40, dry_run=True):
    """
    Remove WAV files that start with silence >= threshold.
    
    Args:
        directory: Directory to scan for WAV files
        silence_threshold: Minimum silence duration in seconds (default 2.0)
        silence_db: dB threshold for silence (default -40dB)
        dry_run: If True, only print files that would be deleted
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    # Find all .wav files
    wav_files = list(directory.glob("*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in '{directory}'")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    print(f"Checking for leading silence >= {silence_threshold}s (threshold: {silence_db}dB)...\n")
    
    files_to_remove = []
    
    for wav_file in wav_files:
        if has_leading_silence(wav_file, silence_threshold, silence_db):
            files_to_remove.append(wav_file)
            print(f"{'[DRY RUN] Would remove' if dry_run else 'Removing'}: {wav_file.name}")
    
    if not files_to_remove:
        print("\nNo files with leading silence found.")
        return
    
    print(f"\n{'Would remove' if dry_run else 'Removed'} {len(files_to_remove)} file(s)")
    
    if not dry_run:
        for wav_file in files_to_remove:
            try:
                wav_file.unlink()
            except Exception as e:
                print(f"Error deleting {wav_file}: {e}")
    else:
        print("\n*** DRY RUN MODE - No files were actually deleted ***")
        print("Set dry_run=False to actually delete files")

# Example usage
if __name__ == "__main__":
    # Configuration
    import sys
    target_directory = sys.argv[1]  # Change to your directory
    silence_threshold_seconds = 2.0     # Minimum silence duration
    silence_db_threshold = -40          # dB threshold for silence
    
    # First run in dry-run mode to see what would be deleted
    remove_silent_wav_files(
        directory=target_directory,
        silence_threshold=silence_threshold_seconds,
        silence_db=silence_db_threshold,
        dry_run=True if sys.argv[2]=='True' else False  # Set to False to actually delete files
    )
