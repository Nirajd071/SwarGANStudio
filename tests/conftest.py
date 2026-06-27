"""Pytest configuration: make the project root importable and provide
shared audio fixtures used across the DSP/model tests.
"""
import os
import sys

import numpy as np
import pytest

# Ensure the project root is importable when pytest is run from elsewhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402


@pytest.fixture
def sine_audio():
    """A 1-second voiced sine tone (220 Hz) with light vibrato."""
    sr = config.SAMPLE_RATE
    t = np.linspace(0, 1.0, sr, endpoint=False)
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5 * t)
    audio = 0.5 * np.sin(2 * np.pi * 220 * t * vibrato)
    return audio.astype(np.float32)
