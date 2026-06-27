"""API integration tests using FastAPI's TestClient."""
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import config
from engine.base import VoiceProfile
from engine.registry import default_registries
from service.app import create_app
from service.config import ServiceSettings


def _wav_bytes(seconds=1.5, sr=config.SAMPLE_RATE, freq=220.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    storage_dir = tmp_path_factory.mktemp("storage")
    ref_path = storage_dir / "ref.wav"
    with open(ref_path, "wb") as f:
        f.write(_wav_bytes(freq=180.0))

    voices, engines = default_registries()
    voices.register(VoiceProfile(
        id="demo", name="Demo Voice", reference_audio=str(ref_path),
        licensed=True, description="A licensed demo voice."))
    voices.register(VoiceProfile(
        id="unlicensed", name="Unlicensed", licensed=False))

    settings = ServiceSettings(storage_dir=str(storage_dir), max_workers=1)
    app = create_app(voices=voices, engines=engines, settings=settings)
    return TestClient(app)


def _poll(client, job_id, timeout=60.0):
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        status = r.json()["status"]
        if status in ("done", "failed"):
            return r.json()
        time.sleep(0.4)
    raise AssertionError(f"Job did not finish in time (last status={status})")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "autovc" in body["engines"]


def test_voices_lists_only_licensed(client):
    r = client.get("/voices")
    assert r.status_code == 200
    ids = [v["id"] for v in r.json()["voices"]]
    assert "demo" in ids
    assert "unlicensed" not in ids


def test_convert_unknown_voice(client):
    r = client.post("/convert",
                    files={"file": ("x.wav", _wav_bytes(), "audio/wav")},
                    data={"voice_id": "ghost"})
    assert r.status_code == 404


def test_convert_unlicensed_voice_forbidden(client):
    r = client.post("/convert",
                    files={"file": ("x.wav", _wav_bytes(), "audio/wav")},
                    data={"voice_id": "unlicensed"})
    assert r.status_code == 403


def test_convert_empty_upload(client):
    r = client.post("/convert",
                    files={"file": ("x.wav", b"", "audio/wav")},
                    data={"voice_id": "demo"})
    assert r.status_code == 400


def test_convert_end_to_end(client):
    r = client.post("/convert",
                    files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"voice_id": "demo"})
    assert r.status_code == 202
    job = r.json()
    assert job["status"] in ("pending", "running")

    final = _poll(client, job["id"])
    assert final["status"] == "done", f"job failed: {final.get('error')}"
    assert final["result_available"] is True

    result = client.get(f"/jobs/{job['id']}/result")
    assert result.status_code == 200
    assert result.headers["content-type"] == "audio/wav"
    assert len(result.content) > 1000


def test_result_not_ready_for_unknown_job(client):
    r = client.get("/jobs/doesnotexist/result")
    assert r.status_code == 404


def test_convert_sync_returns_audio(client):
    r = client.post("/convert/sync",
                    files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"voice_id": "demo"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert len(r.content) > 1000


def test_convert_sync_enforces_licensing(client):
    r = client.post("/convert/sync",
                    files={"file": ("song.wav", _wav_bytes(), "audio/wav")},
                    data={"voice_id": "unlicensed"})
    assert r.status_code == 403


def test_ui_is_served(client):
    r = client.get("/ui/")
    assert r.status_code == 200
    assert "SwarGAN Studio" in r.text


def test_root_redirects_to_ui(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/ui/"
