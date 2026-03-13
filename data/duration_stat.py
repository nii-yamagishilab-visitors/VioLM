import os
import wave
from pathlib import Path

def get_wav_duration(filepath):
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(filepath), 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0

def calculate_total_duration(directory):
    """Calculate total duration of all WAV files in a directory."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    wav_files = list(directory_path.glob("*.wav"))
    
    if not wav_files:
        print(f"No .wav files found in '{directory}'")
        return
    
    total_duration = 0
    file_count = 0
    
    print(f"Found {len(wav_files)} WAV file(s):\n")
    
    for wav_file in wav_files:
        duration = get_wav_duration(wav_file)
        total_duration += duration
        file_count += 1
        
        minutes, seconds = divmod(int(duration), 60)
        # print(f"  {wav_file.name}: {minutes}m {seconds}s ({duration:.2f}s)")
    
    # Convert total duration to hours, minutes, and seconds
    hours, remainder = divmod(int(total_duration), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*50}")
    print(f"Total files: {file_count}")
    print(f"Total duration: {hours}h {minutes}m {seconds}s ({total_duration:.2f}s)")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Specify the directory containing WAV files
    directory = "./"  # Current directory, change as needed
    
    # You can also take directory as command line argument
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    
    calculate_total_duration(directory)