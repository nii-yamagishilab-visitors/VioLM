import pretty_midi
import numpy as np
import os
import argparse
from pathlib import Path

import mido
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm

def has_long_polyphony(midi_file_path, min_notes=2, min_duration=0.1):
    """
    Check if a MIDI file has 3 or more notes played together for longer than 1 second.
    
    Args:
        midi_file_path: Path to MIDI file
        min_notes: Minimum number of simultaneous notes (default: 3)
        min_duration: Minimum duration in seconds (default: 1.0)
        
    Returns:
        True if file should be removed, False otherwise
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_file_path))
        
        for instrument in midi_data.instruments:
            if len(instrument.notes) < min_notes:
                continue
            
            # Create a list of all note events (start and end)
            events = []
            for note in instrument.notes:
                events.append(('start', note.start, note))
                events.append(('end', note.end, note))
            
            # Sort events by time
            events.sort(key=lambda x: x[1])
            
            # Track active notes
            active_notes = []
            
            for event_type, event_time, note in events:
                if event_type == 'start':
                    active_notes.append(note)
                    
                    # Check if we have min_notes or more notes active
                    if len(active_notes) >= min_notes:
                        # Check if any combination of min_notes has been playing together for min_duration
                        # Find the most recent note that started
                        sorted_active = sorted(active_notes, key=lambda n: n.start)
                        
                        # Check if the oldest min_notes notes have been playing together long enough
                        if len(sorted_active) >= min_notes:
                            oldest_notes = sorted_active[:min_notes]
                            latest_start = max(n.start for n in oldest_notes)
                            earliest_end = min(n.end for n in oldest_notes)
                            
                            overlap_duration = earliest_end - latest_start
                            
                            if overlap_duration >= min_duration:
                                return True
                    
                else:  # event_type == 'end'
                    # Remove the note from active notes
                    active_notes = [n for n in active_notes if n != note]
        
        return False
        
    except Exception as e:
        print(f"  Error checking {midi_file_path.name}: {e}")
        return False


def filter_polyphonic_files(directory_path, min_notes=2, min_duration=0.1):
    """
    Filter out MIDI files with sustained polyphony.
    
    Args:
        directory_path: Path to directory containing MIDI files
        min_notes: Minimum number of simultaneous notes
        min_duration: Minimum duration in seconds
        
    Returns:
        List of MIDI files that pass the filter
    """
    midi_files = list(Path(directory_path).glob('*.mid')) + list(Path(directory_path).glob('*.midi'))
    
    print(f"\nFiltering MIDI files...")
    print(f"Removing files with {min_notes}+ notes played together for {min_duration}s or longer")
    
    filtered_files = []
    removed_files = []
    
    for midi_file in tqdm(midi_files, desc="Filtering files"):
        if has_long_polyphony(midi_file, min_notes, min_duration):
            removed_files.append(midi_file.name)
        else:
            filtered_files.append(midi_file)
    
    print(f"\nFiltering complete:")
    print(f"  Total files: {len(midi_files)}")
    print(f"  Kept: {len(filtered_files)}")
    print(f"  Removed: {len(removed_files)}")
    
    for file in removed_files:
        try:
            print("  Removing file:", file)
            os.remove(Path(directory_path) / file)
            print("  Removing file:", file[:-4] + '.wav')
            os.remove(Path(directory_path) / (file[:-4] + '.wav'))  # Also remove corresponding WAV file if exists
        except Exception as e:
            print(f"  Error removing {file}: {e}")
    
    return filtered_files

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter out MIDI files with sustained polyphony.")
    parser.add_argument("--directory", type=str, help="Directory containing MIDI files")
    parser.add_argument("--min_notes", type=int, default=2, help="Minimum number of simultaneous notes (default: 3)")
    parser.add_argument("--min_duration", type=float, default=0.1, help="Minimum duration in seconds (default: 1.0)")
    
    args = parser.parse_args()
    
    filter_polyphonic_files(args.directory, args.min_notes, args.min_duration)