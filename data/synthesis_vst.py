#!/usr/bin/env python3
"""Generate violin audio from MIDI files using REAPER's Python API.

This script:
1. Loads MIDI files from a specified directory
2. Extracts tempo from MIDI and applies it to Reaper project
3. Applies a VST/VST3i plugin (violin instrument)
4. Renders the audio to WAV files
5. Verifies duration matches between MIDI and audio (within 3% tolerance)
6. Deletes audio files with mismatched durations
7. Saves the output files

Requirements:
    - Reaper DAW installed and running
    - reapy library: pip install python-reapy
    - pretty_midi library: pip install pretty_midi
    - librosa library: pip install librosa
    - VST/VST3i plugins installed

Usage:
    python data/synthesis_vst.py --input-dir /path/to/midi/files \\
        --output-dir /path/to/output --plugin "VST3: Violin Plugin Name"
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, List
import logging
import numpy as np
from tqdm import tqdm

from valle.utils.cli import ConsistentArgumentParser

try:
    import reapy
except ImportError:
    reapy = None

import librosa
import pretty_midi


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_silent(audio_path, threshold_db=-60, frame_length=2048, hop_length=512):
    """
    Check if an audio file is silent.
    
    Args:
        audio_path: Path to the audio file
        threshold_db: Threshold in dB below which audio is considered silent (default: -60)
        frame_length: Frame length for RMS calculation
        hop_length: Hop length for RMS calculation
        
    Returns:
        True if the audio is silent, False otherwise
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Check if the audio array is empty or all zeros
        if len(y) == 0 or np.all(y == 0):
            return True
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms)
        
        # Check if maximum RMS is below threshold
        max_db = np.max(rms_db)
        
        return max_db < threshold_db
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return False

class ReaperMIDIRenderer:
    """Handles MIDI to audio rendering in Reaper."""
    
    def __init__(self, plugin_name: Optional[str] = None, sample_rate: int = 44100,
                 fx_chain: Optional[str] = None, track_template: Optional[str] = None,
                 duration_tolerance: float = 0.03):
        """
        Initialize the Reaper MIDI renderer.
        
        Args:
            plugin_name: Name of the VST/VST3i plugin to use
            sample_rate: Sample rate for audio rendering (default: 44100)
            fx_chain: Path to FX chain file (.RfxChain) with pre-configured plugin
            track_template: Path to track template file (.RTrackTemplate) with pre-configured plugin
            duration_tolerance: Maximum allowed duration difference (default: 0.03 = 3%)
        """
        self.plugin_name = plugin_name
        self.sample_rate = sample_rate
        self.fx_chain = fx_chain
        self.track_template = track_template
        self.duration_tolerance = duration_tolerance
        self.project = None
        
    def connect_to_reaper(self):
        """Connect to Reaper instance."""
        if reapy is None:
            logger.error("python-reapy is required: pip install python-reapy")
            return False
        try:
            # Check if Reaper is running and enable ReaScript API
            # logger.info("Connecting to Reaper...")
            self.project = reapy.Project()
            # logger.info("Successfully connected to Reaper")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Reaper: {e}")
            logger.error("Make sure Reaper is running and ReaScript API is enabled.")
            logger.error("In Reaper: Options -> Preferences -> Plug-ins -> ReaScript -> Enable Python for use with ReaScript")
            return False
    
    def _get_midi_tempo_and_duration(self, midi_path: str) -> tuple:
        """
        Extract tempo and duration from MIDI file using pretty_midi.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Tuple of (tempo in BPM, duration in seconds)
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # Get duration
            duration = midi_data.get_end_time()
            
            # Get tempo (BPM)
            # pretty_midi stores tempo changes, we'll use the first one or estimate
            if len(midi_data.get_tempo_changes()[0]) > 0:
                # Get the first tempo
                tempo = midi_data.get_tempo_changes()[1][0]
            else:
                # Estimate tempo if no tempo changes found
                tempo = midi_data.estimate_tempo()
            
            # logger.info(f"MIDI tempo: {tempo:.2f} BPM, duration: {duration:.2f}s")
            return tempo, duration
            
        except Exception as e:
            logger.error(f"Error reading MIDI file with pretty_midi: {e}")
            # Return default tempo and 0 duration on error
            return 120.0, 0.0
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file using librosa.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            logger.error(f"Error reading audio duration: {e}")
            return 0.0
    
    def _verify_duration_match(self, midi_duration: float, audio_duration: float) -> bool:
        """
        Check if MIDI and audio durations match within tolerance.
        
        Args:
            midi_duration: Duration of MIDI file in seconds
            audio_duration: Duration of audio file in seconds
            
        Returns:
            True if durations match within tolerance, False otherwise
        """
        if midi_duration == 0 or audio_duration == 0:
            logger.warning("Invalid duration detected (0 seconds)")
            return False
        
        # Calculate percentage difference
        diff = abs(midi_duration - audio_duration)
        percentage_diff = diff / midi_duration
        
        # logger.info(f"Duration comparison: MIDI={midi_duration:.2f}s, Audio={audio_duration:.2f}s, Diff={percentage_diff*100:.2f}%")
        
        if percentage_diff > self.duration_tolerance:
            logger.warning(f"Duration mismatch exceeds tolerance ({self.duration_tolerance*100}%)")
            logger.info(f"MIDI duration: {midi_duration:.2f}s, audio duration: {audio_duration:.2f}s")
            return False
        
        return True
    
    def _load_fx_chain(self, track, fx_chain_path: str) -> bool:
        """
        Load an FX chain file to a track.
        
        Args:
            track: Reaper track object
            fx_chain_path: Path to .RfxChain file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(fx_chain_path):
            logger.error(f"FX chain file not found: {fx_chain_path}")
            return False
        
        try:
            # Use ReaScript API to load FX chain
            success = reapy.reascript_api.TrackFX_AddByName(
                track.id, fx_chain_path, False, -1000  # -1000 = add from FX chain file
            )
            
            # Check if any FX were added
            if track.n_fxs > 0:
                return True
            else:
                logger.error("No FX were added from chain file")
                return False
                
        except Exception as e:
            logger.error(f"Error loading FX chain: {e}")
            return False
    
    def list_available_plugins(self) -> List[str]:
        """List all available VST/VST3i plugins in Reaper."""
        logger.info("Scanning available plugins...")
        plugins = []
        
        try:
            # Create a temporary track to scan plugins
            track = self.project.add_track()
            
            # Get FX list - this varies by system
            # You can also use RPR_EnumInstalledFX to list installed plugins
            logger.info("Plugins can be listed using Reaper's FX Browser")
            logger.info("Common plugin name formats:")
            logger.info("  - VST3: Plugin Name (Manufacturer)")
            logger.info("  - VST: Plugin Name (Manufacturer)")
            logger.info("  - VSTi: Plugin Name")
            
            # Clean up
            track.delete()
            #self.project.remove_track(track)
            
        except Exception as e:
            logger.warning(f"Could not enumerate plugins: {e}")
        
        return plugins
    
    def process_midi_file(self, midi_path: str, output_path: str, 
                         plugin_name: Optional[str] = None) -> bool:
        """
        Process a single MIDI file and render to audio.
        
        Args:
            midi_path: Path to input MIDI file
            output_path: Path for output WAV file
            plugin_name: Override plugin name for this file
            
        Returns:
            True if successful, False otherwise
        """
        if plugin_name is None:
            plugin_name = self.plugin_name
        
        # Extract tempo and duration from MIDI
        midi_tempo, midi_duration = self._get_midi_tempo_and_duration(midi_path)
        
        if midi_duration == 0:
            logger.error(f"Invalid MIDI duration for file: {midi_path}")
            return False
            
        # logger.info(f"Processing: {midi_path}")
        
        try:
            if self.project is not None:
            # Close existing project and create a new one
            # logger.info("Clearing project...")
            #     self.project = reapy.Project()
            # else:
                # Clear existing tracks
                while len(self.project.tracks) > 0:
                    self.project.tracks[0].delete()
            # Set project tempo
            # logger.info(f"Setting project tempo to {midi_tempo:.2f} BPM")
            reapy.reascript_api.SetCurrentBPM(self.project.id, midi_tempo, False)

            # Add a new track
            # logger.info("Creating MIDI track...")
            track = self.project.add_track()
            track.name = f"MIDI - {Path(midi_path).stem}"
            
            # Import MIDI file to track
            # logger.info(f"Importing MIDI file: {midi_path}")
            reapy.reascript_api.SetEditCurPos(0, True, False)
            # Insert MIDI file as media item
            item = track.add_midi_item(0)  # Start at 0, 10 seconds default length
            
            # Alternative method using ReaScript API directly
            # This imports the MIDI file properly
            import_result = reapy.reascript_api.InsertMedia(midi_path, 0)
            if import_result == 0:
                logger.error(f"Failed to import MIDI file: {midi_path}")
                return False
            
            # Get the imported item and adjust track
            if len(track.items) == 0:
                logger.error("No MIDI items found after import")
                return False
            
            item = track.items[0]
            
            # Load plugin using FX chain, track template, or direct plugin
            if self.fx_chain:
                # Load FX chain (includes plugin with preset/state)
                logger.info(f"Loading FX chain: {self.fx_chain}")
                try:
                    success = self._load_fx_chain(track, self.fx_chain)
                    if not success:
                        logger.error("Failed to load FX chain")
                        return False
                    logger.info("FX chain loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load FX chain: {e}")
                    return False
            elif plugin_name:
                # Load plugin directly
                # logger.info(f"Loading plugin: {plugin_name}")
                try:
                    fx = track.add_fx(plugin_name)
                    # logger.info(f"Plugin loaded successfully")
                    # logger.warning("   Consider using --fx-chain with a pre-configured preset")
                except Exception as e:
                    logger.error(f"Failed to load plugin '{plugin_name}': {e}")
                    # logger.error("Available plugin name formats:")
                    # logger.error("  - 'VST3: Plugin Name (Manufacturer)'")
                    # logger.error("  - 'VST: Plugin Name (Manufacturer)'")
                    # logger.error("  - 'VSTi: Plugin Name'")
                    # logger.error("\nTip: Open Reaper's FX Browser to see exact plugin names")
                    return False
            else:
                logger.warning("No plugin specified - using default MIDI playback")
            
            # Calculate project length based on MIDI content
            project_length = item.position + item.length
            
            # Set project render settings
            # logger.info("Configuring render settings...")
            
            # Set render bounds (time selection)
            self.project.time_selection.start = 0
            self.project.time_selection.end = project_length
            
            # Configure render settings using ReaScript API
            reapy.reascript_api.GetSetProjectInfo_String(
                self.project.id, 'RENDER_FILE', os.path.dirname(output_path), True
            )
            reapy.reascript_api.GetSetProjectInfo_String(
                self.project.id, 'RENDER_PATTERN', os.path.basename(output_path), True
            )
            
            # Set render format to WAV (0x464C4157 = 'WAVE' in hex)
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_FORMAT', 0x20000000, True  # WAV format base
            )
            
            # Set sample rate
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_SRATE', self.sample_rate, True
            )
            
            # Set to render to file
            reapy.reascript_api.GetSetProjectInfo(
                self.project.id, 'RENDER_TAILFLAG', 1, True  # Include tail
            )
            
            # Ensure output directory exists
            # os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Render the project
            # logger.info(f"Rendering audio to: {output_path}")
            # logger.info("This may take a while depending on the MIDI file length...")
            
            # Render using time selection
            time.sleep(2)
            reapy.reascript_api.Main_OnCommand(41824, 0)  # Render project to file
            
            # Wait for render to complete
            # Note: In practice, you may need to poll or use callbacks
            time.sleep(1)  # Basic wait - adjust as needed
            
            # Check if output file was created
            if not os.path.exists(output_path):
                logger.error(f"✗ Render failed - output file not created: {output_path}")
                return False
            
            file_size = os.path.getsize(output_path)
            # logger.info(f"✓ Successfully rendered: {output_path} ({file_size} bytes)")
            
            # Check silence
            if is_silent(output_path):
                # logger.warning(f"Silent audio detected - deleting file: {output_path}")
                try:
                    os.remove(output_path)
                    logger.info(f"✗ Deleted silent audio file")
                except Exception as e:
                    logger.error(f"Failed to delete file: {e}")
                return False

            # Verify duration matches
            audio_duration = self._get_audio_duration(output_path)
            
            if not self._verify_duration_match(midi_duration, audio_duration):
                logger.warning(f"Duration mismatch detected - deleting file: {output_path}")
                try:
                    os.remove(output_path)
                    logger.info(f"✗ Deleted mismatched audio file")
                except Exception as e:
                    logger.error(f"Failed to delete file: {e}")
                return False
            
            # logger.info(f"✓ Duration verification passed")
                
        except Exception as e:
            logger.error(f"Error processing MIDI file: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         plugin_name: Optional[str] = None) -> dict:
        """
        Process all MIDI files in a directory.
        
        Args:
            input_dir: Directory containing MIDI files
            output_dir: Directory for output WAV files
            plugin_name: Plugin to use for rendering
            
        Returns:
            Dictionary with success/failure counts
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Find all MIDI files
        midi_extensions = ['.mid', '.midi', '.MID', '.MIDI']
        midi_files = []
        for ext in midi_extensions:
            midi_files.extend(list(input_path.glob(f'*{ext}')))
            midi_files.extend(list(input_path.glob(f'**/*{ext}')))
        
        # Remove duplicates
        midi_files = list(set(midi_files))
        
        if not midi_files:
            logger.warning(f"No MIDI files found in: {input_dir}")
            return {'success': 0, 'failed': 0, 'total': 0, 'duration_mismatch': 0}
        
        logger.info(f"Found {len(midi_files)} MIDI file(s) to process")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each MIDI file
        results = {
            'success': 0, 
            'failed': 0, 
            'total': len(midi_files),
            'duration_mismatch': 0
        }
        
        for i, midi_file in enumerate(tqdm(midi_files)):
            # logger.info(f"\n{'='*60}")
            # logger.info(f"Processing file {i+1}/{len(midi_files)}")
            # logger.info(f"{'='*60}")
            
            # Generate output filename
            relative_path = midi_file.relative_to(input_path) if midi_file.is_relative_to(input_path) else midi_file
            output_file = output_path / relative_path.with_suffix('.wav')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process the file
            success = self.process_midi_file(
                str(midi_file),
                str(output_file),
                plugin_name
            )
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                # Check if file was deleted due to duration mismatch
                if not os.path.exists(str(output_file)):
                    results['duration_mismatch'] += 1
        
        return results


def main():
    """Main entry point for the script."""
    parser = ConsistentArgumentParser(
        description='Generate violin audio from MIDI files using Reaper with tempo and duration verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process MIDI files with a specific violin plugin
  python data/synthesis_vst.py -i ./midi_files -o ./audio_output \\
      -p "VST3: Violin Instrument (Vendor)"
  
  # Process with custom duration tolerance (5%)
  python data/synthesis_vst.py -i ./midi_files -o ./audio_output \\
      -p "VST3: Violin Instrument (Vendor)" --duration-tolerance 0.05
  
  # Process with default MIDI synth (no plugin)
  python data/synthesis_vst.py -i ./midi_files -o ./audio_output
  
  # List plugin format help
  python data/synthesis_vst.py --list-plugins

Note: Reaper must be running before executing this script!
        """
    )
    
    parser.add_argument(
        '-i', '--input-dir',
        type=str,
        help='Directory containing MIDI files to process'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        help='Directory for output WAV files'
    )
    
    parser.add_argument(
        '-p', '--plugin',
        type=str,
        default=None,
        help='VST/VST3i plugin name as displayed in the REAPER FX browser'
    )

    parser.add_argument(
        '-fx', '--fx-chain',
        type=str,
        default=None,
        help='Path to FX chain file (.RfxChain) with pre-configured plugin and instrument'
    )
    
    parser.add_argument(
        '-tt', '--track-template',
        type=str,
        default=None,
        help='Path to track template file (.RTrackTemplate) with pre-configured setup'
    )
    
    parser.add_argument(
        '-sr', '--sample-rate',
        type=int,
        default=44100,
        help='Sample rate for output audio (default: 44100)'
    )
    
    parser.add_argument(
        '--duration-tolerance',
        type=float,
        default=0.03,
        help='Maximum allowed duration difference as percentage (default: 0.03 = 3%%)'
    )
    
    parser.add_argument(
        '--list-plugins',
        action='store_true',
        help='Show information about plugin naming format and exit'
    )

    args = parser.parse_args()
    
    # Handle --list-plugins
    if args.list_plugins:
        print("\n" + "="*60)
        print("REAPER PLUGIN NAMING GUIDE")
        print("="*60)
        print("\nTo find the exact plugin name:")
        print("1. Open Reaper")
        print("2. Add a track")
        print("3. Click 'FX' button on the track")
        print("4. Browse for your violin plugin")
        print("5. The exact name shown in the FX browser is what you need")
        print("\nCommon formats:")
        print("  VST3: Plugin Name (Manufacturer)")
        print("  VST: Plugin Name (Manufacturer)")
        print("  VSTi: Plugin Name")
        print("\nExample plugin names:")
        print("  - 'VST3: Violin Instrument (Vendor)'")
        print("  - 'VSTi: Instrument Name'")
        print("\nTip: Copy the exact name from Reaper's FX browser")
        print("="*60 + "\n")
        return
    
    # Validate required arguments
    if not args.input_dir or not args.output_dir:
        parser.error("--input-dir and --output-dir are required (or use --list-plugins)")
    
    # Validate FX options
    if args.fx_chain and args.track_template:
        parser.error("Cannot use both --fx-chain and --track-template. Choose one.")
    
    if args.fx_chain and not os.path.exists(args.fx_chain):
        parser.error(f"FX chain file not found: {args.fx_chain}")
    
    if args.track_template and not os.path.exists(args.track_template):
        parser.error(f"Track template file not found: {args.track_template}")
    
    # Validate duration tolerance
    if args.duration_tolerance < 0 or args.duration_tolerance > 1:
        parser.error("Duration tolerance must be between 0 and 1 (0% to 100%)")
    
    # Initialize renderer
    renderer = ReaperMIDIRenderer(
        plugin_name=args.plugin,
        sample_rate=args.sample_rate,
        fx_chain=args.fx_chain,
        track_template=args.track_template,
        duration_tolerance=args.duration_tolerance
    )
    
    # Connect to Reaper
    if not renderer.connect_to_reaper():
        logger.error("Failed to connect to Reaper. Exiting.")
        return 1
    
    # Process directory
    logger.info(f"\nInput directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Duration tolerance: {args.duration_tolerance*100}%")

    if args.fx_chain:
        logger.info(f"FX Chain: {args.fx_chain}")
        logger.info("✓ Using pre-configured FX chain (instrument already loaded)")
    elif args.track_template:
        logger.info(f"Track Template: {args.track_template}")
        logger.info("✓ Using track template (instrument already loaded)")
    elif args.plugin:
        logger.info(f"Plugin: {args.plugin}")
    else:
        logger.warning("No plugin specified - will use default MIDI synth")
    
    logger.info(f"Sample rate: {args.sample_rate} Hz\n")
    
    # Process all files
    results = renderer.process_directory(
        args.input_dir,
        args.output_dir,
        args.plugin
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total files: {results['total']}")
    logger.info(f"Successfully processed: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Duration mismatches (deleted): {results['duration_mismatch']}")
    logger.info("="*60 + "\n")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == '__main__':
    exit(main())
