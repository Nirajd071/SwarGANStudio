"""
Feature extraction utilities for singing voice processing
"""
import numpy as np
import librosa
import pyworld as pw
from typing import Tuple, Optional
import config
from utils.logging_config import get_logger

logger = get_logger(__name__)

class FeatureExtractor:
    """
    Feature extractor for singing voice analysis
    """
    
    def __init__(self, 
                 sr: int = config.SAMPLE_RATE,
                 n_mels: int = config.N_MELS,
                 hop_length: int = config.HOP_LENGTH,
                 win_length: int = config.WIN_LENGTH,
                 n_fft: int = config.N_FFT):
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        
    def _sanitize(self, audio: np.ndarray) -> np.ndarray:
        """Coerce input to a finite 1-D float64 array, padded to at least one
        FFT window so downstream librosa/pyworld calls never fail on short clips.
        """
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float64).reshape(-1))
        if audio.size == 0:
            raise ValueError("Cannot extract features from empty audio.")
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.shape[0] < self.n_fft:
            pad = self.n_fft - audio.shape[0]
            logger.warning("Audio shorter than n_fft (%d); padding %d samples.",
                           self.n_fft, pad)
            audio = np.pad(audio, (0, pad), mode="constant")
        return audio

    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel-spectrogram from audio
        
        Args:
            audio: Audio data array
            
        Returns:
            Mel-spectrogram array (n_mels, n_frames)
        """
        audio = self._sanitize(audio).astype(np.float32)
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_fft=self.n_fft,
            fmin=config.F_MIN,
            fmax=config.F_MAX
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec
    
    def extract_f0_pyworld(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 (pitch) using PyWorld.

        Args:
            audio: Audio data array

        Returns:
            Tuple of (log_f0, voiced_flag). log_f0 is 0 on unvoiced frames.
        """
        log_f0, voiced_flag, _, _, _ = self._extract_f0_world(audio)
        return log_f0, voiced_flag

    def _extract_f0_world(self, audio: np.ndarray):
        """Run WORLD's DIO + StoneMask once and return everything callers need.

        Returns:
            (log_f0, voiced_flag, f0_linear, time_axis, audio_double)
        """
        audio_double = self._sanitize(audio)

        # Extract F0 using the DIO algorithm, then refine with StoneMask
        f0_linear, time_axis = pw.dio(
            audio_double, self.sr,
            frame_period=self.hop_length / self.sr * 1000)
        f0_linear = pw.stonemask(audio_double, f0_linear, time_axis, self.sr)

        voiced_flag = f0_linear > 0

        # Log-transform F0 (only for voiced frames; unvoiced stay 0)
        log_f0 = np.zeros_like(f0_linear)
        log_f0[voiced_flag] = np.log(f0_linear[voiced_flag])

        return log_f0, voiced_flag, f0_linear, time_axis, audio_double
    
    def extract_spectral_envelope(self, audio: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        Extract spectral envelope using WORLD
        
        Args:
            audio: Audio data array
            f0: F0 contour
            
        Returns:
            Spectral envelope
        """
        # Convert to double precision
        audio_double = self._sanitize(audio)
        f0 = np.ascontiguousarray(f0, dtype=np.float64)

        # Time axis for WORLD
        time_axis = np.arange(len(f0)) * self.hop_length / self.sr
        
        # Extract spectral envelope
        sp = pw.cheaptrick(audio_double, f0, time_axis, self.sr)
        
        return sp
    
    def extract_aperiodicity(self, audio: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """
        Extract aperiodicity using WORLD
        
        Args:
            audio: Audio data array
            f0: F0 contour
            
        Returns:
            Aperiodicity
        """
        # Convert to double precision
        audio_double = self._sanitize(audio)
        f0 = np.ascontiguousarray(f0, dtype=np.float64)

        # Time axis for WORLD
        time_axis = np.arange(len(f0)) * self.hop_length / self.sr
        
        # Extract aperiodicity
        ap = pw.d4c(audio_double, f0, time_axis, self.sr)
        
        return ap
    
    def normalize_features(self, features: np.ndarray, 
                          mean: Optional[np.ndarray] = None,
                          std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize features to zero mean and unit variance
        
        Args:
            features: Feature array
            mean: Pre-computed mean (optional)
            std: Pre-computed std (optional)
            
        Returns:
            Tuple of (normalized_features, mean, std)
        """
        if mean is None:
            mean = np.mean(features, axis=-1, keepdims=True)
        if std is None:
            std = np.std(features, axis=-1, keepdims=True)
            # Prevent division by zero
            std = np.where(std == 0, 1.0, std)
        
        normalized = (features - mean) / std
        return normalized, mean, std
    
    def denormalize_features(self, normalized_features: np.ndarray,
                           mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """
        Denormalize features
        
        Args:
            normalized_features: Normalized feature array
            mean: Mean used for normalization
            std: Std used for normalization
            
        Returns:
            Denormalized features
        """
        return normalized_features * std + mean
    
    def extract_all_features(self, audio: np.ndarray) -> dict:
        """
        Extract all features from audio.

        Args:
            audio: Audio data array

        Returns:
            Dictionary containing all extracted features.
        """
        # Mel-spectrogram (handles its own sanitization)
        mel_spec = self.extract_mel_spectrogram(audio)

        # F0 + spectral envelope + aperiodicity share a single WORLD analysis,
        # so the linear F0 and time axis are consistent across all three.
        log_f0, voiced_flag, f0_linear, time_axis, audio_double = \
            self._extract_f0_world(audio)

        f0_linear = np.ascontiguousarray(f0_linear, dtype=np.float64)
        sp = pw.cheaptrick(audio_double, f0_linear, time_axis, self.sr)
        ap = pw.d4c(audio_double, f0_linear, time_axis, self.sr)

        logger.info("Extracted features: mel=%s, frames=%d, voiced=%.0f%%",
                    mel_spec.shape, len(log_f0), 100.0 * float(np.mean(voiced_flag)))

        return {
            'mel_spec': mel_spec,
            'f0': log_f0,
            'voiced_flag': voiced_flag,
            'spectral_envelope': sp,
            'aperiodicity': ap,
        }
