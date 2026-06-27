"""Tests for audio utility / validation functions."""
import numpy as np
import pytest

import config
from utils.audio_utils import (
    validate_audio,
    normalize_loudness,
    load_audio,
    save_audio,
    MIN_AUDIO_SAMPLES,
)


def test_validate_audio_rejects_empty():
    with pytest.raises(ValueError):
        validate_audio(np.array([]), "empty")


def test_validate_audio_rejects_silent():
    with pytest.raises(ValueError):
        validate_audio(np.zeros(config.SAMPLE_RATE, dtype=np.float32), "silent")


def test_validate_audio_rejects_too_short():
    with pytest.raises(ValueError):
        validate_audio(np.full(MIN_AUDIO_SAMPLES - 1, 0.1, dtype=np.float32), "short")


def test_validate_audio_downmixes_stereo_and_sanitizes_nan():
    base = np.sin(np.linspace(0, 50, config.SAMPLE_RATE)).astype(np.float32)
    stereo = np.stack([base, base])
    stereo[0, 10] = np.nan
    out = validate_audio(stereo, "stereo")
    assert out.ndim == 1
    assert out.shape[0] == config.SAMPLE_RATE
    assert np.isfinite(out).all()


def test_normalize_loudness_no_clipping():
    audio = np.sin(np.linspace(0, 100, 10000)).astype(np.float32)
    out = normalize_loudness(audio)
    assert np.max(np.abs(out)) <= 0.95 + 1e-6


def test_normalize_loudness_silent_passthrough():
    silent = np.zeros(1000, dtype=np.float32)
    out = normalize_loudness(silent)
    assert np.allclose(out, silent)


def test_load_save_roundtrip(tmp_path, sine_audio):
    path = tmp_path / "tone.wav"
    save_audio(sine_audio, str(path), config.SAMPLE_RATE)
    assert path.exists()
    loaded, sr = load_audio(str(path), sr=config.SAMPLE_RATE)
    assert sr == config.SAMPLE_RATE
    assert loaded.ndim == 1
    assert len(loaded) == pytest.approx(len(sine_audio), abs=2)


def test_load_audio_missing_file():
    with pytest.raises(ValueError):
        load_audio("/nonexistent/path/to/audio.wav")
