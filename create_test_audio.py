#!/usr/bin/env python3
"""
Create test audio files for testing SwarGAN audio upload functionality
"""
import numpy as np
import soundfile as sf

def create_test_audio():
    """Create two test WAV files with sine waves"""
    sr = 22050  # Sample rate
    duration = 1  # 1 second
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Create source audio: 440 Hz sine wave (A4 note)
    freq_source = 440
    source_wave = 0.2 * np.sin(2 * np.pi * freq_source * t)
    sf.write('/tmp/source.wav', source_wave.astype(np.float32), sr)
    
    # Create target audio: 330 Hz sine wave (E4 note)
    freq_target = 330
    target_wave = 0.2 * np.sin(2 * np.pi * freq_target * t)
    sf.write('/tmp/target.wav', target_wave.astype(np.float32), sr)
    
    print("Created test audio files:")
    print("- /tmp/source.wav (440 Hz, 1s)")
    print("- /tmp/target.wav (330 Hz, 1s)")

if __name__ == "__main__":
    create_test_audio()