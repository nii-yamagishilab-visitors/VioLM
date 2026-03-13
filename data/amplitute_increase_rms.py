import os
import numpy as np
import librosa
import soundfile as sf

def calculate_rms(audio):
    """Calculate the RMS (Root Mean Square) of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def amplify_to_target_rms(audio, target_rms_increase):
    """
    Amplify audio signal to increase its RMS by a specific amount.
    
    Parameters:
    - audio: numpy array of audio samples
    - target_rms_increase: desired increase in RMS value
    
    Returns:
    - amplified audio signal
    """
    current_rms = calculate_rms(audio)
    target_rms = current_rms + target_rms_increase
    
    # Calculate the gain factor needed
    if current_rms > 0:
        gain = target_rms / current_rms
    else:
        gain = 1.0
    
    # Apply gain to the audio
    amplified_audio = audio * gain
    
    return amplified_audio, current_rms, target_rms

def process_directory(input_dir, output_dir, rms_increase=0.2):
    """
    Process all .wav files in a directory to increase their RMS.
    
    Parameters:
    - input_dir: directory containing input .wav files
    - output_dir: directory to save processed files
    - rms_increase: amount to increase RMS by (default 0.2)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all .wav files in the input directory
    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
    
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return
    
    print(f"Found {len(wav_files)} .wav files to process")
    print(f"Target RMS increase: {rms_increase}\n")
    
    for filename in wav_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Load audio file
            audio, sr = librosa.load(input_path, sr=None, mono=False)
            
            # Process audio (handle both mono and stereo)
            if audio.ndim == 1:
                # Mono audio
                amplified_audio, orig_rms, new_rms = amplify_to_target_rms(audio, rms_increase)
            else:
                # Stereo or multi-channel audio
                amplified_audio = np.zeros_like(audio)
                orig_rms_list = []
                new_rms_list = []
                
                for i in range(audio.shape[0]):
                    amplified_audio[i], orig_rms, new_rms = amplify_to_target_rms(audio[i], rms_increase)
                    orig_rms_list.append(orig_rms)
                    new_rms_list.append(new_rms)
                
                orig_rms = np.mean(orig_rms_list)
                new_rms = np.mean(new_rms_list)
            
            # Save the amplified audio
            sf.write(output_path, amplified_audio.T if audio.ndim > 1 else amplified_audio, sr)
            
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}\n")
    
    print("Processing complete!")

if __name__ == "__main__":
    # Configuration
    import sys
    INPUT_DIR = sys.argv[1]#"./test_increase"  # Change this to your input directory
    OUTPUT_DIR = sys.argv[2]#"./output_increase"  # Change this to your output directory
    RMS_INCREASE = float(sys.argc[3])#0.2  # Amount to increase RMS by
    
    process_directory(INPUT_DIR, OUTPUT_DIR, RMS_INCREASE)
