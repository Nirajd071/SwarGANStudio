"""
Audio utility functions for SwarGAN
"""
import numpy as np
import librosa
import soundfile as sf
import torch
from typing import Tuple, Optional
import config

def load_audio(file_path: str, sr: int = config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        # Normalize audio
        audio = librosa.util.normalize(audio)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")

def save_audio(audio: np.ndarray, file_path: str, sr: int = config.SAMPLE_RATE):
    """
    Save audio array to file
    
    Args:
        audio: Audio data array
        file_path: Output file path
        sr: Sample rate
    """
    try:
        sf.write(file_path, audio, sr)
    except Exception as e:
        raise ValueError(f"Error saving audio file {file_path}: {str(e)}")

def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Audio data array
        top_db: Threshold for silence detection
        
    Returns:
        Trimmed audio array
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed

def normalize_loudness(audio: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
    """
    Normalize audio loudness using RMS approximation
    
    Args:
        audio: Audio data array
        target_lufs: Target loudness in LUFS
        
    Returns:
        Normalized audio array
    """
    # Simple RMS-based normalization (approximation of LUFS)
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        # Convert target LUFS to linear scale (approximation)
        target_rms = 10**(target_lufs / 20)
        scaling_factor = target_rms / rms
        normalized_audio = audio * scaling_factor
        # Clip to prevent distortion
        normalized_audio = np.clip(normalized_audio, -0.95, 0.95)
        return normalized_audio
    return audio

def split_audio_frames(audio: np.ndarray, frame_length: int, hop_length: int) -> list:
    """
    Split audio into overlapping frames
    
    Args:
        audio: Audio data array
        frame_length: Length of each frame
        hop_length: Hop length between frames
        
    Returns:
        List of audio frames
    """
    frames = []
    for start in range(0, len(audio) - frame_length + 1, hop_length):
        frame = audio[start:start + frame_length]
        frames.append(frame)
    return frames

def audio_to_tensor(audio: np.ndarray) -> torch.Tensor:
    """
    Convert numpy audio array to PyTorch tensor
    
    Args:
        audio: Audio data array
        
    Returns:
        PyTorch tensor
    """
    return torch.from_numpy(audio).float()

def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy audio array
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Audio data array
    """
    return tensor.detach().cpu().numpy()
