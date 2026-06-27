"""
Reference conversion engine backed by the in-repo AutoVC model + vocoder.

This is intentionally the *baseline* backend. It runs end-to-end today and
defines the contract a stronger backend (RVC / so-vits-svc / diffusion) will
later satisfy. Audio quality depends on a trained checkpoint (see README).
"""
from typing import Dict, Optional, Tuple

import numpy as np
import librosa

import config
from engine.base import ConversionEngine, VoiceProfile
from utils.audio_utils import load_audio
from utils.logging_config import get_logger

logger = get_logger(__name__)


class AutoVCEngine(ConversionEngine):
    name = "autovc"

    def __init__(self):
        self._processor = None
        self._voice_feature_cache: Dict[str, dict] = {}

    def load(self) -> None:
        if self._processor is None:
            # Imported lazily so importing the engine module stays cheap.
            from audio_processor import AudioProcessor
            logger.info("Initializing AutoVC engine.")
            self._processor = AudioProcessor()

    def _voice_features(self, voice: VoiceProfile) -> Optional[dict]:
        """Compute (and cache) the target mel features for a voice profile."""
        if voice.reference_audio is None:
            return None
        if voice.id in self._voice_feature_cache:
            return self._voice_feature_cache[voice.id]

        ref_audio, _ = load_audio(voice.reference_audio, sr=config.SAMPLE_RATE)
        mel = self._processor.feature_extractor.extract_mel_spectrogram(ref_audio)
        features = {"mel_spec": mel}
        self._voice_feature_cache[voice.id] = features
        return features

    def convert(
        self,
        source_audio: np.ndarray,
        sr: int,
        voice: VoiceProfile,
    ) -> Tuple[np.ndarray, int]:
        self.load()

        source_audio = np.ascontiguousarray(
            np.asarray(source_audio, dtype=np.float32).reshape(-1))

        # The AutoVC model operates at config.SAMPLE_RATE.
        if sr != config.SAMPLE_RATE:
            source_audio = librosa.resample(
                source_audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)

        target_features = self._voice_features(voice)
        converted = self._processor.convert_voice(source_audio, target_features)
        return np.asarray(converted, dtype=np.float32), config.SAMPLE_RATE
