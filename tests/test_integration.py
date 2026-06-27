"""End-to-end integration test: audio -> features -> AutoVC -> waveform."""
import numpy as np
import torch

import config
from utils.feature_extractor import FeatureExtractor
from models.autovc import AutoVC
from models.vocoder import create_vocoder


def test_full_conversion_pipeline(sine_audio):
    fe = FeatureExtractor()
    model = AutoVC().eval()
    vocoder = create_vocoder("griffin_lim")

    # Use the sine tone as both source content and target style.
    mel = fe.extract_mel_spectrogram(sine_audio)
    mel_t = torch.from_numpy(mel).float().unsqueeze(0)

    with torch.no_grad():
        out = model(mel_t, mel_t)
        wav = vocoder(out["converted_postnet"]).squeeze().numpy()

    assert wav.ndim == 1
    assert wav.shape[0] > 0
    assert np.isfinite(wav).all()
