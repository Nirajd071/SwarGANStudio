"""Tests for the feature extraction pipeline."""
import numpy as np
import pytest

import config
from utils.feature_extractor import FeatureExtractor


@pytest.fixture
def fe():
    return FeatureExtractor()


def test_mel_spectrogram_shape(fe, sine_audio):
    mel = fe.extract_mel_spectrogram(sine_audio)
    assert mel.shape[0] == config.N_MELS
    assert mel.shape[1] > 0


def test_f0_voiced_alignment(fe, sine_audio):
    log_f0, voiced = fe.extract_f0_pyworld(sine_audio)
    assert log_f0.shape == voiced.shape
    # A 220 Hz tone should be mostly voiced
    assert np.mean(voiced) > 0.5
    # Voiced frames carry a sensible pitch (~220 Hz)
    f0_hz = np.exp(log_f0[voiced])
    assert 100 < np.median(f0_hz) < 400


def test_all_features_frame_consistency(fe, sine_audio):
    feats = fe.extract_all_features(sine_audio)
    n = len(feats["f0"])
    assert feats["voiced_flag"].shape[0] == n
    assert feats["spectral_envelope"].shape[0] == n
    assert feats["aperiodicity"].shape[0] == n
    assert feats["mel_spec"].shape[0] == config.N_MELS


def test_short_clip_is_padded(fe):
    short = np.full(200, 0.1, dtype=np.float32)
    mel = fe.extract_mel_spectrogram(short)
    assert mel.shape[0] == config.N_MELS
    assert mel.shape[1] >= 1


def test_empty_audio_raises(fe):
    with pytest.raises(ValueError):
        fe.extract_mel_spectrogram(np.array([], dtype=np.float32))


def test_normalize_denormalize_roundtrip(fe):
    x = np.random.randn(config.N_MELS, 50).astype(np.float32)
    norm, mean, std = fe.normalize_features(x)
    recovered = fe.denormalize_features(norm, mean, std)
    assert np.allclose(recovered, x, atol=1e-4)
