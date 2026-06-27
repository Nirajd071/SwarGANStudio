"""Tests for the prebuilt SVC backend command templates."""
from engine.external_engine import ExternalCommandEngine
from engine.svc_templates import (
    make_so_vits_svc_engine,
    make_rvc_engine,
    make_python_script_engine,
)


def _render(engine, source="in.wav", output="out.wav"):
    return engine._render({
        "source": source, "output": output, "target": "",
        "voice_id": "rafi", "model": "",
    })


def test_so_vits_svc_template():
    eng = make_so_vits_svc_engine(
        model_path="G_rafi.pth", config_path="config.json", speaker="rafi")
    assert isinstance(eng, ExternalCommandEngine)
    assert eng.name == "so-vits-svc"
    cmd = _render(eng)
    assert cmd[0] == "svc" and cmd[1] == "infer"
    assert "G_rafi.pth" in cmd and "config.json" in cmd and "rafi" in cmd
    assert "in.wav" in cmd and "out.wav" in cmd


def test_rvc_template_with_index():
    eng = make_rvc_engine(model_path="rafi.pth", index_path="rafi.index",
                          f0_method="rmvpe", transpose=2)
    assert eng.name == "rvc"
    cmd = _render(eng)
    assert "rafi.pth" in cmd
    assert "rafi.index" in cmd
    assert "rmvpe" in cmd
    assert "2" in cmd  # transpose
    assert "in.wav" in cmd and "out.wav" in cmd


def test_rvc_template_without_index_omits_flag():
    eng = make_rvc_engine(model_path="rafi.pth")
    cmd = _render(eng)
    assert "-ip" not in cmd


def test_python_script_template_extra_args():
    eng = make_python_script_engine(
        script_path="infer.py", name="myssvc",
        extra_args=["--speaker", "{voice_id}", "--model", "{model}"])
    assert eng.name == "myssvc"
    cmd = eng._render({
        "source": "a.wav", "output": "b.wav", "target": "",
        "voice_id": "kishore", "model": "m.pth",
    })
    assert "infer.py" in cmd
    assert "kishore" in cmd
    assert "m.pth" in cmd


def test_template_engines_register_in_registry():
    from engine.registry import EngineRegistry
    reg = EngineRegistry()
    reg.register(make_rvc_engine(model_path="x.pth"))
    assert "rvc" in reg.names()
    assert reg.get("rvc").name == "rvc"
