"""
Audio utility functions for SwarGAN
"""
import os
import numpy as np
import librosa
import soundfile as sf
import torch
from typing import Tuple
import config
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Minimum number of samples we consider a usable clip (~50 ms at 22.05 kHz).
MIN_AUDIO_SAMPLES = 1024
# Threshold below which a signal is considered effectively silent.
SILENCE_RMS_THRESHOLD = 1e-5


def validate_audio(audio: np.ndarray, source: str = "audio") -> np.ndarray:
    """Validate and sanitize a raw audio array.

    Handles the real-world edge cases the pipeline is likely to hit:
    empty arrays, NaN/Inf samples, multi-channel input, silent clips and
    clips that are too short to process.

    Args:
        audio: Audio samples (1-D, or multi-channel that will be downmixed).
        source: Human-readable label used in error/log messages.

    Returns:
        A clean 1-D float32 mono array.

    Raises:
        ValueError: If the audio is empty, silent, or too short to use.
    """
    if audio is None or np.size(audio) == 0:
        raise ValueError(f"{source}: audio is empty.")

    audio = np.asarray(audio, dtype=np.float32)

    # Downmix to mono if multi-channel
    if audio.ndim > 1:
        logger.warning("%s: received %d channels, downmixing to mono.",
                       source, audio.shape[0] if audio.shape[0] < audio.shape[-1] else audio.shape[-1])
        audio = np.mean(audio, axis=0 if audio.shape[0] < audio.shape[-1] else -1)

    audio = np.ascontiguousarray(audio.reshape(-1))

    # Replace NaN/Inf with zeros rather than letting them poison the pipeline
    if not np.isfinite(audio).all():
        n_bad = int(np.sum(~np.isfinite(audio)))
        logger.warning("%s: %d non-finite samples replaced with 0.", source, n_bad)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    if audio.shape[0] < MIN_AUDIO_SAMPLES:
        raise ValueError(
            f"{source}: clip too short ({audio.shape[0]} samples, "
            f"minimum {MIN_AUDIO_SAMPLES}).")

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < SILENCE_RMS_THRESHOLD:
        raise ValueError(f"{source}: clip appears to be silent (rms={rms:.2e}).")

    # Warn (but don't fail) on clipped input
    peak = float(np.max(np.abs(audio)))
    if peak >= 0.999:
        logger.warning("%s: input may be clipped (peak amplitude %.3f).", source, peak)

    return audio


def load_audio(file_path: str, sr: int = config.SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file
        sr: Target sample rate

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Audio file not found: {file_path}")

    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")

    # Validate / sanitize before any normalization
    audio = validate_audio(audio, source=os.path.basename(file_path))

    # Peak-normalize (validate_audio already guaranteed a non-silent signal)
    audio = librosa.util.normalize(audio)
    logger.info("Loaded '%s': %.2fs @ %d Hz", os.path.basename(file_path),
                len(audio) / sr, sr)
    return audio, sr

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
