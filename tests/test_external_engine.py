"""Tests for the pluggable external (subprocess) conversion engine.

These use a tiny fake CLI tool (a Python one-liner) to exercise the
round-trip without depending on a real GPU SVC stack.
"""
import sys

import numpy as np
import pytest

from engine.base import VoiceProfile
from engine.external_engine import ExternalCommandEngine

import config

# A fake "SVC tool": reads {source}, applies a trivial transform, writes {output}.
_FAKE_TOOL = (
    "import sys, soundfile as sf, numpy as np;"
    "a, sr = sf.read(sys.argv[1], dtype='float32');"
    "sf.write(sys.argv[2], (a * 0.5).astype('float32'), sr)"
)

_FAILING_TOOL = "import sys; sys.exit(3)"


def _source():
    sr = config.SAMPLE_RATE
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr


def test_external_engine_roundtrip():
    engine = ExternalCommandEngine(
        command_template=[sys.executable, "-c", _FAKE_TOOL, "{source}", "{output}"],
        name="fake-svc",
    )
    assert engine.name == "fake-svc"

    source, sr = _source()
    voice = VoiceProfile(id="demo", name="Demo", licensed=True)
    out, out_sr = engine.convert(source, sr, voice)

    assert out_sr == sr
    assert out.ndim == 1
    assert np.isfinite(out).all()
    # The fake tool halves amplitude
    assert np.max(np.abs(out)) < np.max(np.abs(source))


def test_external_engine_nonzero_exit_raises():
    engine = ExternalCommandEngine(
        command_template=[sys.executable, "-c", _FAILING_TOOL, "{source}", "{output}"],
        name="broken",
    )
    source, sr = _source()
    voice = VoiceProfile(id="demo", name="Demo", licensed=True)
    with pytest.raises(RuntimeError):
        engine.convert(source, sr, voice)


def test_external_engine_missing_output_raises():
    # Tool exits 0 but writes nothing.
    engine = ExternalCommandEngine(
        command_template=[sys.executable, "-c", "pass", "{source}", "{output}"],
        name="noop",
    )
    source, sr = _source()
    voice = VoiceProfile(id="demo", name="Demo", licensed=True)
    with pytest.raises(RuntimeError):
        engine.convert(source, sr, voice)


def test_external_engine_rejects_empty_template():
    with pytest.raises(ValueError):
        ExternalCommandEngine(command_template=[])


def test_external_engine_template_placeholders_filled():
    engine = ExternalCommandEngine(
        command_template=["tool", "--in", "{source}", "--out", "{output}",
                          "--voice", "{voice_id}", "--model", "{model}"],
        name="t",
    )
    rendered = engine._render({
        "source": "s.wav", "output": "o.wav", "target": "",
        "voice_id": "rafi", "model": "m.pth",
    })
    assert rendered == ["tool", "--in", "s.wav", "--out", "o.wav",
                        "--voice", "rafi", "--model", "m.pth"]
