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

    return filtered_files

def extract_velocities_from_directory(directory_path):
    """
    Extract all note velocities from MIDI files in a directory.
    
    Args:
        directory_path: Path to directory containing MIDI files
        
    Returns:
        List of all velocities found in the MIDI files
    """
    velocities = []
    midi_files = list(Path(directory_path).glob('*.mid')) + list(Path(directory_path).glob('*.midi'))
    
    print(f"Found {len(midi_files)} MIDI files in source directory")
    
    for midi_file in tqdm(midi_files, 'Extracting Velocity Information'):
        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_file))
            
            # Extract velocities from all instruments
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    velocities.append(note.velocity)
                    
        except Exception as e:
            print(f"  Error processing {midi_file.name}: {e}")
    
    return velocities


def calculate_velocity_distribution(velocities):
    """
    Calculate the velocity distribution from a list of velocities.
    
    Args:
        velocities: List of velocity values
        
    Returns:
        Numpy array of velocities and statistics dictionary
    """
    velocities_array = np.array(velocities)
    
    stats = {
        'mean': np.mean(velocities_array),
        'std': np.std(velocities_array),
        'min': np.min(velocities_array),
        'max': np.max(velocities_array),
        'median': np.median(velocities_array)
    }
    
    print(f"\nVelocity Distribution Statistics:")
    print(f"  Total notes: {len(velocities_array)}")
    print(f"  Min velocity: {stats['min']}")
    print(f"  Max velocity: {stats['max']}")
    print(f"  Mean velocity: {stats['mean']:.2f}")
    print(f"  Std deviation: {stats['std']:.2f}")
    print(f"  Median velocity: {stats['median']:.2f}")
    
    return velocities_array, stats


def sample_velocity_from_distribution(velocities, prev_velocity=None, max_change=15):
    """
    Sample a velocity from the distribution with smooth transitions.
    
    Args:
        velocities: Array of velocities from source distribution
        prev_velocity: Previous velocity value for smooth transition
        max_change: Maximum allowed change from previous velocity
        
    Returns:
        New velocity value
    """
    if prev_velocity is None:
        # First note: sample randomly from distribution
        return int(np.random.choice(velocities))
    
    # Sample a velocity from the distribution
    candidate = int(np.random.choice(velocities))
    
    # Apply smoothing constraint: limit the change from previous velocity
    change = candidate - prev_velocity
    
    if abs(change) > max_change:
        # If change is too large, limit it
        if change > 0:
            candidate = prev_velocity + max_change
        else:
            candidate = prev_velocity - max_change
    
    # Ensure valid MIDI range
    candidate = max(1, min(127, candidate))
    
    return candidate


def group_simultaneous_notes(notes):
    """
    Group notes that start and end at the same time (double stops, chords).
    
    Args:
        notes: List of notes sorted by start time
        
    Returns:
        List of lists, where each inner list contains simultaneous notes
    """
    if len(notes) == 0:
        return []
    
    groups = []
    current_group = [notes[0]]
    
    for i in range(1, len(notes)):
        # Check if this note starts and ends at the same time as the previous note
        if (abs(notes[i].start - current_group[0].start) < 0.01 and 
            abs(notes[i].end - current_group[0].end) < 0.01):
            current_group.append(notes[i])
        else:
            groups.append(current_group)
            current_group = [notes[i]]
    
    groups.append(current_group)
    return groups


def apply_velocity_distribution(midi_file_path, velocity_distribution, output_path, max_change=15):
    """
    Apply velocity distribution to a MIDI file with smooth local transitions.
    Notes that start and end at the same time (double stops) get the same velocity.
    
    Args:
        midi_file_path: Path to input MIDI file
        velocity_distribution: Array of velocities from source
        output_path: Path to save modified MIDI file
        max_change: Maximum velocity change between consecutive notes
    """
    midi_data = pretty_midi.PrettyMIDI(str(midi_file_path))
    
    for instrument in midi_data.instruments:
        if len(instrument.notes) == 0:
            continue
        
        # Sort notes by start time to maintain musical sequence
        notes_sorted = sorted(instrument.notes, key=lambda x: x.start)
        
        # Group simultaneous notes (double stops, chords)
        note_groups = group_simultaneous_notes(notes_sorted)
        
        # Apply velocities with smooth transitions
        prev_velocity = None
        for group in note_groups:
            # Sample one velocity for the entire group
            new_velocity = sample_velocity_from_distribution(
                velocity_distribution, 
                prev_velocity, 
                max_change
            )
            
            # Apply the same velocity to all notes in the group
            for note in group:
                note.velocity = new_velocity
            
            prev_velocity = new_velocity
    
    # Save modified MIDI file
    midi_data.write(str(output_path))


def add_velocity_directory(target_directory, velocity_distribution, output_directory, filtered_files=None, max_change=15):
    """
    Process all MIDI files in target directory with the velocity distribution.
    
    Args:
        target_directory: Path to directory containing MIDI files to process
        velocity_distribution: Array of velocities from source
        output_directory: Path to output directory
        max_change: Maximum velocity change between consecutive notes
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    midi_files = list(Path(target_directory).glob('*.mid')) + list(Path(target_directory).glob('*.midi'))
    if filtered_files:
        filtered_paths = set(Path(f).resolve() for f in filtered_files)
        # Remove filtered files from midi_files
        midi_files = [f for f in midi_files if f.resolve() in filtered_paths]
    
    print(f"\nProcessing {len(midi_files)} MIDI files...")
    print(f"Applying velocity distribution with max change of {max_change} between notes...")
    
    for midi_file in tqdm(midi_files, "Adding Velocity"):
        output_file = output_path / midi_file.name
        apply_velocity_distribution(midi_file, velocity_distribution, output_file, max_change)
    
    print(f"\nCompleted! Processed files saved to: {output_directory}")

def add_articulation_keyswitches(input_file, output_file, tempo=500000, 
                                 staccato_threshold=0.25,
                                 tempo_threshold = 80, 
                                 staccato_keyswitch=24,
                                 legato_keyswitch=28,
                                 detache_keyswitch=25,
                                 harmonic_keyswitch=32,
                                 pizz_keyswitch=26):
    """
    Add articulation keyswitches to a MIDI file based on note duration.
    
    Parameters:
    - input_file: Path to input MIDI file
    - output_file: Path to output MIDI file
    - tempo: Tempo in microseconds per beat (default 500000 = 120 BPM)
    - staccato_threshold: Duration threshold in quarter notes (default 0.25 = 1/16 note)
    - staccato_keyswitch: MIDI note number for staccato keyswitch (default C1 = 24)
    - legato_keyswitch: MIDI note number for legato keyswitch (default C#1 = 25)
    """
    
    # Load the MIDI file
    mid = MidiFile(input_file)

    pretty_midi_obj = pretty_midi.PrettyMIDI(input_file)

    keyswitch_dict = {'staccato': staccato_keyswitch,
                       'legato': legato_keyswitch,
                       'detache': detache_keyswitch,
                       'harmonic': harmonic_keyswitch,
                       'pizz': pizz_keyswitch}
    tempo = 0
    if len(pretty_midi_obj.get_tempo_changes()[0]) > 0:
        # Get the first tempo
        tempo = pretty_midi_obj.get_tempo_changes()[1][0]
    else:
        # Estimate tempo if no tempo changes found
        tempo = pretty_midi_obj.estimate_tempo()
    
    # Get ticks per beat
    ticks_per_beat = mid.ticks_per_beat
    
    # Calculate threshold in ticks
    threshold_ticks = staccato_threshold * ticks_per_beat
    
    # Create new MIDI file
    new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
    
    for i, track in enumerate(mid.tracks):
        # First pass: collect all events with absolute timing
        events = []
        current_time = 0
        note = 0
        for msg in track:
            current_time += msg.time
            if msg.is_meta:
                if msg.type != 'end_of_track':
                    events.append({
                        'time': current_time,
                        'type': 'meta',
                        'msg': msg.copy(time=0)
                    })
            elif msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'time': current_time,
                    'type': 'note_on',
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                })
                note += 1
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({
                    'time': current_time,
                    'type': 'note_off',
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                })
                note += 1
            else:
                events.append({
                    'time': current_time,
                    'type': 'other',
                    'msg': msg.copy(time=0)
                })
        
        # Second pass: match note_on with note_off and determine articulation
        note_pairs = []
        active_notes = {}
        
        for event in events:
            if event['type'] == 'note_on':
                active_notes[event['note']] = event
            elif event['type'] == 'note_off':
                if event['note'] in active_notes:
                    note_on = active_notes[event['note']]
                    duration = event['time'] - note_on['time']
                    
                    # articulation = 'staccato' if (duration < threshold_ticks and tempo > tempo_threshold) else 'legato'
                    articulation = 'detache'
                    note_pairs.append({
                        'note_on_time': note_on['time'],
                        'note_off_time': event['time'],
                        'note': event['note'],
                        'velocity': note_on['velocity'],
                        'channel': note_on['channel'],
                        'articulation': articulation
                    })
                    
                    del active_notes[event['note']]
        
        # Third pass: rebuild track with keyswitches
        all_events = []
        
        # Add meta messages
        for event in events:
            if event['type'] == 'meta':
                all_events.append(event)
        
        # Add keyswitches and notes
        current_articulation = None
        for pair in note_pairs:
            # Add keyswitch if needed
            if pair['articulation'] != current_articulation:
                keyswitch_note = keyswitch_dict[pair['articulation']]
                
                # Keyswitch on
                all_events.append({
                    'time': pair['note_on_time'],
                    'type': 'keyswitch_on',
                    'note': keyswitch_note,
                    'channel': pair['channel']
                })
                
                # Keyswitch off (immediately after)
                all_events.append({
                    'time': pair['note_on_time'],
                    'type': 'keyswitch_off',
                    'note': keyswitch_note,
                    'channel': pair['channel']
                })
                
                current_articulation = pair['articulation']
            
            # Add note on
            all_events.append({
                'time': pair['note_on_time'],
                'type': 'note_on',
                'note': pair['note'],
                'velocity': pair['velocity'],
                'channel': pair['channel']
            })
            
            # Add note off
            all_events.append({
                'time': pair['note_off_time'],
                'type': 'note_off',
                'note': pair['note'],
                'velocity': 0,
                'channel': pair['channel']
            })
        
        # Add other events
        for event in events:
            if event['type'] == 'other':
                all_events.append(event)
        
        # Sort all events by time
        all_events.sort(key=lambda x: x['time'])
        
        if note == 0:
            continue
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)

        # Convert absolute times to delta times and add to track
        last_time = 0
        for event in all_events:
            delta_time = event['time'] - last_time
            
            if event['type'] == 'meta':
                new_track.append(event['msg'].copy(time=delta_time))
            elif event['type'] == 'note_on':
                new_track.append(Message('note_on', note=event['note'], 
                                       velocity=event['velocity'], 
                                       time=delta_time, 
                                       channel=event['channel']))
            elif event['type'] == 'note_off':
                new_track.append(Message('note_off', note=event['note'], 
                                       velocity=event['velocity'], 
                                       time=delta_time, 
                                       channel=event['channel']))
            elif event['type'] == 'keyswitch_on':
                new_track.append(Message('note_on', note=event['note'], 
                                       velocity=127, 
                                       time=delta_time, 
                                       channel=event['channel']))
            elif event['type'] == 'keyswitch_off':
                new_track.append(Message('note_off', note=event['note'], 
                                       velocity=0, 
                                       time=delta_time, 
                                       channel=event['channel']))
            elif event['type'] == 'other':
                new_track.append(event['msg'].copy(time=delta_time))
            
            last_time = event['time']
        
        # Add end of track
        new_track.append(mido.MetaMessage('end_of_track', time=0))
    
    # Save the new MIDI file
    new_mid.save(output_file)

def add_articulation_directory(target_directory, output_directory, key_switch_config={}):
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    # Threshold: 0.25 = 1/16 note, 0.5 = 1/8 note, etc.
    STACCATO_THRESHOLD = 0.5 if 'STACCATO_THRESHOLD' not in key_switch_config else key_switch_config['STACCATO_THRESHOLD']
    THRESHOLD_SPEED = 80 if 'THRESHOLD_SPEED' not in key_switch_config else key_switch_config['THRESHOLD_SPEED']
    
    # Keyswitch MIDI note numbers (adjust based on your sample library)
    STACCATO_KEYSWITCH = 24 if 'STACCATO_KEYSWITCH' not in key_switch_config else key_switch_config['STACCATO_KEYSWITCH']  # C1
    LEGATO_KEYSWITCH = 28 if 'LEGATO_KEYSWITCH' not in key_switch_config else key_switch_config['LEGATO_KEYSWITCH']   # C#1
    DETACHE_KEYSWITCH = 25 if 'LEGATO_KEYSWITCH' not in key_switch_config else key_switch_config['LEGATO_KEYSWITCH']   # C#1
    HARMONIC_KEYSWITCH = 32 if 'LEGATO_KEYSWITCH' not in key_switch_config else key_switch_config['LEGATO_KEYSWITCH']   # C#1
    PIZZ_KEYSWITCH = 26 if 'LEGATO_KEYSWITCH' not in key_switch_config else key_switch_config['LEGATO_KEYSWITCH']   # C#1

    midi_files = list(Path(target_directory).glob('*.mid')) + list(Path(target_directory).glob('*.midi'))
    
    print(f"Found {len(midi_files)} MIDI files in source directory")
    
    for midi_file in tqdm(midi_files, "Adding articulation"):
        output_file = Path(output_directory) / midi_file.name
        add_articulation_keyswitches(
            midi_file, 
            output_file,
            staccato_threshold=STACCATO_THRESHOLD,
            tempo_threshold=THRESHOLD_SPEED,
            staccato_keyswitch=STACCATO_KEYSWITCH,
            legato_keyswitch=LEGATO_KEYSWITCH,
            detache_keyswitch=DETACHE_KEYSWITCH,
            harmonic_keyswitch=HARMONIC_KEYSWITCH,
            pizz_keyswitch=PIZZ_KEYSWITCH
        )



def add_cc_control(input_file, output_file, 
                   staccato_threshold=0.25,
                   tempo_threshold=80,
                   cc_config={}):
    """
    Add CC control messages to a MIDI file based on note articulation.
    
    Parameters:
    - input_file: Path to input MIDI file
    - output_file: Path to output MIDI file
    - staccato_threshold: Duration threshold in quarter notes (default 0.25 = 1/16 note)
    - tempo_threshold: Tempo threshold in BPM (default 100)
    - cc_channel: CC channel number to use (default 3)
    - staccato_cc_value: CC value for staccato articulation (default 100)
    - legato_cc_value: CC value for legato articulation (default 1)
    """
    
    # Load the MIDI file
    mid = MidiFile(input_file)
    
    try:
        pretty_midi_obj = pretty_midi.PrettyMIDI(str(input_file))
        
        # Get tempo
        tempo = 120  # default
        if len(pretty_midi_obj.get_tempo_changes()[0]) > 0:
            tempo = pretty_midi_obj.get_tempo_changes()[1][0]
        else:
            tempo = pretty_midi_obj.estimate_tempo()
    except:
        tempo = 120  # fallback
    
    # Get ticks per beat
    ticks_per_beat = mid.ticks_per_beat
    
    # Calculate threshold in ticks
    threshold_ticks = staccato_threshold * ticks_per_beat
    
    # print(f"  Processing with tempo={tempo:.1f} BPM, threshold={staccato_threshold} quarters, threshold_ticks={threshold_ticks}")
    
    # Create new MIDI file
    new_mid = MidiFile(ticks_per_beat=ticks_per_beat)
    for track_idx, track in enumerate(mid.tracks):
        # First pass: collect all events with absolute timing
        events = []
        current_time = 0
        note = 0
        for msg in track:
            current_time += msg.time
            
            if msg.is_meta:
                events.append({
                    'time': current_time,
                    'type': 'meta',
                    'msg': msg.copy(time=0)
                })
            elif msg.type == 'note_on' and msg.velocity > 0:
                events.append({
                    'time': current_time,
                    'type': 'note_on',
                    'note': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel
                })
                note += 1
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                events.append({
                    'time': current_time,
                    'type': 'note_off',
                    'note': msg.note,
                    'velocity': msg.velocity if msg.type == 'note_off' else 0,
                    'channel': msg.channel
                })
                note += 1
            else:
                # Keep all other messages (CC, pitch bend, etc.)
                events.append({
                    'time': current_time,
                    'type': 'other',
                    'msg': msg.copy(time=0)
                })
        # Second pass: match note_on with note_off and determine articulation
        note_pairs = []
        active_notes = {}  # key: (note, channel)
        #print(tempo, tempo_threshold, threshold_ticks)
        
        for event in events:
            if event['type'] == 'note_on':
                key = (event['note'], event['channel'])
                active_notes[key] = event
            elif event['type'] == 'note_off':
                key = (event['note'], event['channel'])
                if key in active_notes:
                    note_on = active_notes[key]
                    duration = event['time'] - note_on['time']
                    # Determine articulation
                    # articulation = 'staccato' if (duration < threshold_ticks and tempo > tempo_threshold) else 'legato'
                    articulation = 'harmonic'
                    note_pairs.append({
                        'note_on_time': note_on['time'],
                        'note_off_time': event['time'],
                        'note': event['note'],
                        'velocity': note_on['velocity'],
                        'channel': note_on['channel'],
                        'articulation': articulation,
                        'duration_ticks': duration
                    })
                    
                    del active_notes[key]
        
        #print(f"    Track {track_idx}: Found {len(note_pairs)} note pairs")
        
        # Count articulations
        staccato_count = sum(1 for p in note_pairs if p['articulation'] == 'staccato')
        legato_count = sum(1 for p in note_pairs if p['articulation'] == 'legato')
        #print(f"    Staccato: {staccato_count}, Legato: {legato_count}")
        
        # Third pass: rebuild track with CC messages
        all_events = []
        
        # Add all original events first
        all_events.extend(events)
        # Add Empression CC Control for SWAM VIOLIN
        if note != 0:
            all_events.append({
                'time': 1,
                'type': 'cc_message',
                'cc_channel': 11,  
                'cc_value': 96,   
                'channel': 0      
            })

        # Add CC messages based on articulation changes
        if len(note_pairs) > 0:
            current_articulation = None
            
            for pair in note_pairs:
                # Add CC message if articulation changed
                if pair['articulation'] != current_articulation:
                    # Insert CC message just before the note_on
                    all_events.append({
                        'time': pair['note_on_time'] - 1,  # Slightly before the note
                        'type': 'cc_message',
                        'cc_channel': cc_config[pair['articulation']]['CC_CHANNEL'],
                        'cc_value': cc_config[pair['articulation']]['value'],
                        'channel': pair['channel']
                    })
                    if pair['articulation'] == 'staccato':
                        all_events.append({
                            'time': pair['note_on_time'] - 1,  # Slightly before the note
                            'type': 'cc_message',
                            'cc_channel': cc_config['off_string']['CC_CHANNEL'],
                            'cc_value': cc_config['off_string']['value'],
                            'channel': pair['channel']
                        })
                    else:
                        all_events.append({
                            'time': pair['note_on_time'] - 1,  # Slightly before the note
                            'type': 'cc_message',
                            'cc_channel': cc_config['on_string']['CC_CHANNEL'],
                            'cc_value': cc_config['on_string']['value'],
                            'channel': pair['channel']
                        })
                    current_articulation = pair['articulation']
        
        # Sort all events by time, then by type priority
        def event_sort_key(event):
            # CC messages should come before notes at the same time
            type_priority = {
                'meta': 0,
                'cc_message': 1,
                'note_on': 2,
                'note_off': 3,
                'other': 4
            }
            return (event['time'], type_priority.get(event['type'], 99))
        
        all_events.sort(key=event_sort_key)
        
        # Convert absolute times to delta times and add to track
        last_time = 0
        cc_added = 0
        
        if note == 0:
            continue

        new_track = MidiTrack()
        new_mid.tracks.append(new_track)

        for event in all_events:
            delta_time = max(0, event['time'] - last_time)
            
            if event['type'] == 'meta':
                new_track.append(event['msg'].copy(time=delta_time))
            elif event['type'] == 'note_on':
                new_track.append(Message('note_on', 
                                       note=event['note'], 
                                       velocity=event['velocity'], 
                                       time=delta_time, 
                                       channel=event['channel']))
            elif event['type'] == 'note_off':
                new_track.append(Message('note_off', 
                                       note=event['note'], 
                                       velocity=event['velocity'], 
                                       time=delta_time, 
                                       channel=event['channel']))
            elif event['type'] == 'cc_message':
                new_track.append(Message('control_change', 
                                       control=event['cc_channel'], 
                                       value=event['cc_value'],
                                       time=delta_time, 
                                       channel=event['channel']))
                cc_added += 1
            elif event['type'] == 'other':
                new_track.append(event['msg'].copy(time=delta_time))
            
            last_time = event['time']
        
        #print(f"    Added {cc_added} CC messages to track {track_idx}")
    
    # Save the new MIDI file
    new_mid.save(output_file)
    #print(f"  Saved: {output_file}")


def add_cc_directory(target_directory, output_directory, cc_config={}):
    """
    Process all MIDI files in target directory, adding CC control based on articulation.
    
    Args:
        target_directory: Path to directory containing MIDI files to process
        output_directory: Path to output directory
        cc_config: Dictionary containing CC configuration parameters
    """
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get configuration with defaults
    STACCATO_THRESHOLD = cc_config.get('STACCATO_THRESHOLD', 0.5)
    THRESHOLD_SPEED = cc_config.get('THRESHOLD_SPEED', 80)
    HARMONIC_CC_CHANNEL = cc_config.get('CC_CHANNEL', 14)
    PLAYMODE_CC_CHANNEL = cc_config.get('CC_CHANNEL', 3)
    SUSTAIN_CC_CHANNEL = cc_config.get('CC_CHANNEL', 64)
    BOWLIFT_CC_CHANNEL = cc_config.get('CC_CHANNEL', 15)
    STACCATO_CC_VALUE = cc_config.get('STACCATO_CC_VALUE', 0)
    OFF_STRING_CC_VALUE = cc_config.get('STACCATO_LIFT_CC_VALUE', 127)
    ON_STRING_CC_VALUE = cc_config.get('STACCATO_LIFT_CC_VALUE', 0)
    PIZZ_CC_VALUE = cc_config.get('PIZZ_CC_VALUE', 64)
    DETACHE_CC_VALUE = cc_config.get('DETACHE_CC_VALUE', 127)
    HARMONIC_CC_VALUE = cc_config.get('HARMONIC_CC_VALUE', 42)
    LEGATO_CC_VALUE = cc_config.get('LEGATO_CC_VALUE', 0)

    midi_files = list(Path(target_directory).glob('*.mid')) + list(Path(target_directory).glob('*.midi'))

    cc_config = {
        'harmonic': {'CC_CHANNEL': HARMONIC_CC_CHANNEL, 'value': HARMONIC_CC_VALUE},
        'staccato': {'CC_CHANNEL': SUSTAIN_CC_CHANNEL, 'value': STACCATO_CC_VALUE},
        'off_string': {'CC_CHANNEL': BOWLIFT_CC_CHANNEL, 'value': OFF_STRING_CC_VALUE},
        'on_string': {'CC_CHANNEL': BOWLIFT_CC_CHANNEL, 'value': ON_STRING_CC_VALUE},
        'legato': {'CC_CHANNEL': SUSTAIN_CC_CHANNEL, 'value': LEGATO_CC_VALUE},
        'pizzicato': {'CC_CHANNEL': PLAYMODE_CC_CHANNEL, 'value': PIZZ_CC_VALUE},
        'detache': {'CC_CHANNEL': SUSTAIN_CC_CHANNEL, 'value': DETACHE_CC_VALUE},
    }
    
    for midi_file in tqdm(midi_files, desc="Adding CC Control"):
        try:
            output_file = Path(output_directory) / midi_file.name
            add_cc_control(
                str(midi_file), 
                str(output_file),
                staccato_threshold=STACCATO_THRESHOLD,
                tempo_threshold=THRESHOLD_SPEED,
                cc_config = cc_config
            )
        except Exception as e:
            print(f"\n  Error processing {midi_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nCompleted! Processed files saved to: {output_directory}")

def main():
    parser = argparse.ArgumentParser(
        description='Apply velocity distribution from source MIDI files to target MIDI files with smooth local transitions.'
    )
    parser.add_argument('--reference_dir', type=str, help='Directory containing source MIDI files for analysis')
    parser.add_argument('--target_dir', type=str, help='Directory containing target MIDI files to process')
    parser.add_argument('--output_dir', type=str, help='Directory to save processed MIDI files')
    parser.add_argument(
        '--max-change',
        type=int,
        default=15,
        help='Maximum velocity change between consecutive notes (default: 15)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MIDI Velocity Distribution Matcher")
    print("=" * 70)
    
    # Step 0: Filter out files with sustained polyphony from target directory
    print(f"\nStep 0: Filtering target directory: {args.target_dir}")
    filtered_files = filter_polyphonic_files(args.target_dir, min_notes=2, min_duration=0.1)
    print(len(filtered_files))
    if len(filtered_files) == 0:
        print("Error: No files remaining after filtering!")
        return
    # Step 1: Extract velocities from source directory
    print(f"\nStep 1: Analyzing source directory: {args.reference_dir}")
    velocities = extract_velocities_from_directory(args.reference_dir)
    
    if len(velocities) == 0:
        print("Error: No velocities found in source directory!")
        return
    
    # Step 2: Calculate velocity distribution
    print("\nStep 2: Calculating velocity distribution...")
    velocity_distribution, stats = calculate_velocity_distribution(velocities)
    
    # Step 3: Apply to target directory
    print(f"\nStep 3: Applying to target directory: {args.target_dir}")
    print(f"Each note will be assigned a random velocity from the source distribution")
    print(f"Velocity changes between consecutive notes limited to ±{args.max_change}")
    add_velocity_directory(args.target_dir, velocity_distribution, 'tmp_midi_directory_1', max_change=args.max_change, filtered_files=filtered_files)

    # # Step 4: Add Articulation
    add_articulation_directory(target_directory='tmp_midi_directory_1', output_directory=args.output_dir)

    # Step 5: Add CC Control
    # add_cc_directory(target_directory='tmp_midi_directory_1', output_directory=args.output_dir)

    # Clear temporary directory
    import shutil
    shutil.rmtree('tmp_midi_directory_1')
    
    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
