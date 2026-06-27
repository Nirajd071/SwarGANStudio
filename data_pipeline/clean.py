"""
Cleaning stage: turn a raw (already vocal-separated) waveform into a
normalized, de-rumbled, edge-trimmed mono signal at the dataset sample rate.
"""
import os
from typing import Union

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

from data_pipeline.config import PipelineConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


def load_audio_file(path: str, sr: int) -> np.ndarray:
    """Load an audio file as mono float32 at the target sample rate."""
    if not path or not os.path.exists(path):
        raise ValueError(f"Audio file not found: {path}")
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


def remove_dc(audio: np.ndarray) -> np.ndarray:
    """Remove any DC offset."""
    return audio - float(np.mean(audio))


def highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Apply a zero-phase Butterworth high-pass filter to remove rumble.

    Falls back to a no-op for signals too short to filter.
    """
    if cutoff_hz <= 0 or audio.shape[0] < 27:  # filtfilt needs some padding room
        return audio
    nyq = 0.5 * sr
    norm = min(cutoff_hz / nyq, 0.99)
    sos = butter(4, norm, btype="highpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def peak_normalize(audio: np.ndarray, target: float) -> np.ndarray:
    """Scale so the peak amplitude equals ``target`` (no-op for silence)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-9:
        return audio
    return (audio * (target / peak)).astype(np.float32)


def trim_edges(audio: np.ndarray, top_db: float) -> np.ndarray:
    """Trim leading/trailing silence."""
    if audio.size == 0:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed if trimmed.size > 0 else audio


def clean_vocal(source: Union[str, np.ndarray], cfg: PipelineConfig) -> np.ndarray:
    """Run the full cleaning chain on a vocal stem.

    Args:
        source: Path to an audio file, or a raw waveform array.
        cfg: Pipeline configuration.

    Returns:
        Cleaned mono float32 waveform at ``cfg.sample_rate``.
    """
    if isinstance(source, str):
        audio = load_audio_file(source, cfg.sample_rate)
        label = os.path.basename(source)
    else:
        audio = np.ascontiguousarray(np.asarray(source, dtype=np.float32).reshape(-1))
        label = "array"

    if audio.size == 0:
        raise ValueError(f"{label}: empty audio.")

    # Replace any non-finite samples
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

    audio = remove_dc(audio)
    audio = highpass(audio, cfg.sample_rate, cfg.highpass_hz)
    audio = trim_edges(audio, cfg.trim_top_db)
    audio = peak_normalize(audio, cfg.peak_target)

    logger.debug("Cleaned %s: %d samples (%.2fs)",
                 label, audio.shape[0], audio.shape[0] / cfg.sample_rate)
    return audio
