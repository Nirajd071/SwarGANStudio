"""Tests for the AutoVC model, loss, and vocoders."""
import numpy as np
import pytest
import torch

import config
from models.autovc import AutoVC, AutoVCLoss, SpeakerEncoder, ContentEncoder
from models.vocoder import create_vocoder, SimpleVocoder


@pytest.fixture(scope="module")
def model():
    return AutoVC().eval()


def test_speaker_encoder_output_shape():
    enc = SpeakerEncoder()
    x = torch.randn(3, config.N_MELS, 120)
    emb = enc(x)
    assert emb.shape == (3, config.STYLE_DIM)
    # Embeddings are L2-normalized
    norms = torch.linalg.norm(emb, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_content_encoder_output_shape():
    enc = ContentEncoder()
    x = torch.randn(2, config.N_MELS, 100)
    code = enc(x)
    assert code.shape == (2, 100, config.BOTTLENECK_DIM)


@pytest.mark.parametrize("batch,T", [(1, 101), (2, 128), (4, 64)])
def test_autovc_forward_shapes(model, batch, T):
    x = torch.randn(batch, config.N_MELS, T)
    out = model(x, x)
    assert out["converted"].shape == (batch, config.N_MELS, T)
    assert out["converted_postnet"].shape == (batch, config.N_MELS, T)
    assert out["content"].shape == (batch, T, config.BOTTLENECK_DIM)
    assert out["content_recon"].shape == out["content"].shape
    assert out["speaker_emb"].shape == (batch, config.STYLE_DIM)


def test_autovc_reconstruction_path_without_target(model):
    x = torch.randn(2, config.N_MELS, 80)
    out = model(x)  # no target -> uses source speaker embedding
    assert out["converted_postnet"].shape == x.shape


def test_autovc_loss_and_backward():
    m = AutoVC()
    x = torch.randn(2, config.N_MELS, 96)
    out = m(x, x)
    loss_dict = AutoVCLoss()(out, x)
    for key in ("total_loss", "rec_loss", "rec_psnt_loss", "content_loss"):
        assert key in loss_dict
        assert torch.isfinite(loss_dict[key])
    loss_dict["total_loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_griffin_lim_vocoder_output(model):
    voc = create_vocoder("griffin_lim")
    assert isinstance(voc, SimpleVocoder)
    # Batched input -> (B, L) output
    mel = torch.randn(1, config.N_MELS, 100)
    wav = voc(mel)
    assert wav.dim() == 2 and wav.shape[0] == 1
    assert torch.isfinite(wav).all()
    assert wav.shape[-1] > config.HOP_LENGTH * 50

    # Unbatched input (n_mels, T) -> (L,) output
    wav_1d = voc(torch.randn(config.N_MELS, 100))
    assert wav_1d.dim() == 1


def test_mel_to_linear_is_nonnegative_magnitude():
    voc = SimpleVocoder()
    mel = torch.randn(1, config.N_MELS, 40) * 10 - 20  # dB-ish range
    lin = voc.mel_to_linear(mel)
    assert lin.shape[1] == config.N_FFT // 2 + 1
    assert (lin >= 0).all()


def test_neural_vocoder_shape():
    voc = create_vocoder("neural")
    mel = torch.randn(2, config.N_MELS, 50)
    wav = voc(mel)
    assert wav.shape[0] == 2
    assert wav.shape[1] == 1


def test_create_vocoder_invalid():
    with pytest.raises(ValueError):
        create_vocoder("does_not_exist")
