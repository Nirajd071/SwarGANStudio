"""Tests for the conversion-engine abstraction and registries."""
import numpy as np
import pytest
import soundfile as sf

import config
from engine.base import VoiceProfile, ConversionEngine
from engine.registry import VoiceRegistry, EngineRegistry, default_registries
from engine.autovc_engine import AutoVCEngine


def _sine_wav(path, seconds=1.5, sr=config.SAMPLE_RATE, freq=220.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return str(path)


def test_voice_profile_public_dict_hides_internal_fields():
    v = VoiceProfile(id="x", name="X", reference_audio="/secret/path.wav",
                     licensed=True, metadata={"model": "secret"})
    pub = v.to_public_dict()
    assert "reference_audio" not in pub
    assert "metadata" not in pub
    assert pub["licensed"] is True


def test_voice_registry_register_and_filter():
    reg = VoiceRegistry()
    reg.register(VoiceProfile(id="a", name="A", licensed=True))
    reg.register(VoiceProfile(id="b", name="B", licensed=False))
    assert reg.get("a").name == "A"
    assert len(reg.list()) == 2
    assert [v.id for v in reg.list_licensed()] == ["a"]


def test_voice_registry_load_from_directory(tmp_path):
    _sine_wav(tmp_path / "rafi.wav")
    _sine_wav(tmp_path / "kishore.wav")
    reg = VoiceRegistry()
    n = reg.load_from_directory(str(tmp_path), licensed=True)
    assert n == 2
    assert reg.get("rafi").licensed is True


def test_engine_registry_default_and_lookup():
    eng = EngineRegistry()
    eng.register(AutoVCEngine(), default=True)
    assert "autovc" in eng.names()
    assert eng.get().name == "autovc"
    assert eng.get("autovc").name == "autovc"
    with pytest.raises(KeyError):
        eng.get("nonexistent")


def test_default_registries_have_autovc():
    voices, engines = default_registries()
    assert engines.get().name == "autovc"
    assert voices.list() == []


def test_autovc_engine_converts(tmp_path):
    ref = _sine_wav(tmp_path / "ref.wav", freq=180.0)
    voice = VoiceProfile(id="demo", name="Demo", reference_audio=ref, licensed=True)
    engine = AutoVCEngine()

    sr = config.SAMPLE_RATE
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    source = (0.5 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)

    out, out_sr = engine.convert(source, sr, voice)
    assert out_sr == config.SAMPLE_RATE
    assert out.ndim == 1
    assert out.shape[0] > 0
    assert np.isfinite(out).all()
