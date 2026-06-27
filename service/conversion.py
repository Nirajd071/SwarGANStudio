"""
Conversion orchestration: the product flow behind a job.

    load -> (optional) separate vocals -> convert -> (optional) remix -> save
"""
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import librosa

import config
from engine.base import ConversionEngine, VoiceProfile
from utils.audio_utils import load_audio
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _try_separate(input_path: str):
    """Attempt Demucs separation. Returns (vocals_path, instrumental_path) or
    (None, None) if Demucs/ffmpeg are unavailable or separation fails."""
    try:
        from data_pipeline.ingest import separate_vocals, demucs_available
        if not demucs_available():
            logger.warning("Demucs unavailable; skipping separation.")
            return None, None
        out_dir = tempfile.mkdtemp(prefix="swargan_sep_")
        vocals = separate_vocals(input_path, out_dir)
        instrumental = os.path.join(os.path.dirname(vocals), "no_vocals.wav")
        return vocals, (instrumental if os.path.exists(instrumental) else None)
    except Exception as e:  # pragma: no cover - depends on optional tooling
        logger.warning("Separation failed (%s); using full mix as vocals.", e)
        return None, None


def _remix(vocals: np.ndarray, instrumental_path: str, sr: int) -> np.ndarray:
    """Mix converted vocals back with the instrumental stem."""
    inst, _ = librosa.load(instrumental_path, sr=sr, mono=True)
    n = max(len(vocals), len(inst))
    vocals = np.pad(vocals, (0, n - len(vocals)))
    inst = np.pad(inst, (0, n - len(inst)))
    mix = 0.9 * vocals + 0.9 * inst
    peak = float(np.max(np.abs(mix))) or 1.0
    if peak > 0.99:
        mix = mix * (0.99 / peak)
    return mix.astype(np.float32)


def run_conversion(
    input_path: str,
    voice: VoiceProfile,
    engine: ConversionEngine,
    output_path: str,
    separate: bool = False,
) -> str:
    """Run the full conversion flow and write the result to ``output_path``.

    Returns the output path.
    """
    logger.info("Conversion job: voice=%s engine=%s separate=%s",
                voice.id, engine.name, separate)

    audio, sr = load_audio(input_path, sr=config.SAMPLE_RATE)

    instrumental_path: Optional[str] = None
    vocals = audio
    if separate:
        voc_path, instrumental_path = _try_separate(input_path)
        if voc_path is not None:
            vocals, sr = load_audio(voc_path, sr=config.SAMPLE_RATE)

    converted, out_sr = engine.convert(vocals, sr, voice)

    if instrumental_path is not None:
        converted = _remix(converted, instrumental_path, out_sr)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, converted.astype(np.float32), out_sr)
    logger.info("Conversion complete -> %s (%d samples @ %d Hz)",
                output_path, len(converted), out_sr)
    return output_path
