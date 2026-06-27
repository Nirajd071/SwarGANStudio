"""Tests for the singer data pipeline (clean / segment / manifest / orchestrator)."""
import json
import os

import numpy as np
import pytest
import soundfile as sf

from data_pipeline.config import PipelineConfig
from data_pipeline.clean import (
    clean_vocal, remove_dc, highpass, peak_normalize, load_audio_file,
)
from data_pipeline.segment import (
    compute_quality, segment_audio, accept_segment, segment_and_filter,
    SegmentQuality,
)
from data_pipeline.manifest import (
    write_segments, write_manifest, read_manifest, summarize, write_summary,
)
from data_pipeline.pipeline import build_singer_dataset


# Small, fast config for tests.
@pytest.fixture
def cfg():
    return PipelineConfig(
        sample_rate=16000,
        segment_seconds=2.0,
        min_segment_seconds=1.0,
        max_segment_seconds=4.0,
    )


def _voiced(sr, seconds, freq=220.0, amp=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------- clean.py ---------------------------------------

def test_remove_dc():
    x = np.ones(100, dtype=np.float32) * 0.3
    assert abs(float(np.mean(remove_dc(x)))) < 1e-6


def test_peak_normalize_targets_peak():
    x = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = peak_normalize(x, 0.95)
    assert np.isclose(np.max(np.abs(out)), 0.95, atol=1e-5)


def test_peak_normalize_silence_passthrough():
    x = np.zeros(50, dtype=np.float32)
    assert np.allclose(peak_normalize(x, 0.95), x)


def test_highpass_attenuates_dc_offset(cfg):
    x = _voiced(cfg.sample_rate, 1.0) + 0.5  # strong DC component
    out = highpass(x, cfg.sample_rate, cfg.highpass_hz)
    assert abs(float(np.mean(out))) < abs(float(np.mean(x)))


def test_clean_vocal_from_array(cfg):
    audio = _voiced(cfg.sample_rate, 2.0) + 0.4
    out = clean_vocal(audio, cfg)
    assert out.dtype == np.float32
    assert np.isclose(np.max(np.abs(out)), cfg.peak_target, atol=1e-3)


def test_clean_vocal_empty_raises(cfg):
    with pytest.raises(ValueError):
        clean_vocal(np.array([], dtype=np.float32), cfg)


def test_load_audio_missing(cfg):
    with pytest.raises(ValueError):
        load_audio_file("/no/such/file.wav", cfg.sample_rate)


# --------------------------- segment.py -------------------------------------

def test_compute_quality_voiced(cfg):
    q = compute_quality(_voiced(cfg.sample_rate, 2.0), cfg.sample_rate)
    assert q.voiced_ratio > 0.5
    assert 150 < q.f0_median_hz < 300
    assert q.rms > 0.1


def test_compute_quality_silence(cfg):
    q = compute_quality(np.zeros(cfg.sample_rate * 2, dtype=np.float32), cfg.sample_rate)
    assert q.voiced_ratio == 0.0
    assert q.rms < 1e-6


def test_segment_audio_splits_long_voiced(cfg):
    # 4s voiced + 1.5s silence + 4s voiced
    sr = cfg.sample_rate
    audio = np.concatenate([
        _voiced(sr, 4.0),
        np.zeros(int(sr * 1.5), dtype=np.float32),
        _voiced(sr, 4.0, freq=180.0),
    ])
    segs = segment_audio(audio, sr, cfg)
    # Each 4s voiced region splits into 2x ~2s chunks -> ~4 segments
    assert len(segs) >= 3
    for s in segs:
        assert s.shape[0] >= int(cfg.min_segment_seconds * sr)
        assert s.shape[0] <= int(cfg.max_segment_seconds * sr)


def test_accept_segment_rejections(cfg):
    base = dict(duration=3.0, rms=0.2, peak=0.9, voiced_ratio=0.8,
                f0_median_hz=220.0, f0_mean_hz=220.0)

    assert accept_segment(SegmentQuality(**base), cfg)[0] is True

    too_short = {**base, "duration": 0.5}
    assert accept_segment(SegmentQuality(**too_short), cfg) == (False, "too_short")

    too_quiet = {**base, "rms": 0.0001}
    assert accept_segment(SegmentQuality(**too_quiet), cfg) == (False, "too_quiet")

    low_voiced = {**base, "voiced_ratio": 0.05}
    assert accept_segment(SegmentQuality(**low_voiced), cfg) == (False, "low_voiced_ratio")

    bad_pitch = {**base, "f0_median_hz": 5000.0}
    assert accept_segment(SegmentQuality(**bad_pitch), cfg) == (False, "pitch_out_of_range")


def test_segment_and_filter_keeps_voiced(cfg):
    sr = cfg.sample_rate
    audio = _voiced(sr, 4.0)
    accepted, rejects = segment_and_filter(audio, sr, cfg)
    assert len(accepted) >= 1
    for _, q in accepted:
        assert q.voiced_ratio >= cfg.min_voiced_ratio


# --------------------------- manifest.py ------------------------------------

def test_manifest_write_read_and_summary(cfg, tmp_path):
    sr = cfg.sample_rate
    accepted, _ = segment_and_filter(_voiced(sr, 4.0), sr, cfg)
    assert accepted, "expected at least one accepted segment"

    out_dir = tmp_path / "rafi"
    entries = write_segments(accepted, "rafi", "song1.wav", str(out_dir), cfg)
    assert len(entries) == len(accepted)
    for e in entries:
        assert os.path.exists(e.path)

    manifest_path = tmp_path / "rafi.manifest.jsonl"
    write_manifest(entries, str(manifest_path))
    rows = read_manifest(str(manifest_path))
    assert len(rows) == len(entries)
    assert rows[0]["singer"] == "rafi"

    summary = summarize(entries)
    assert summary["num_segments"] == len(entries)
    assert "rafi" in summary["singers"]
    assert summary["total_duration_s"] > 0


def test_summarize_empty():
    assert summarize([])["num_segments"] == 0


# --------------------------- pipeline.py ------------------------------------

def test_build_singer_dataset_end_to_end(cfg, tmp_path):
    sr = cfg.sample_rate
    # Two synthetic "songs" of voiced audio
    song_paths = []
    for i in range(2):
        p = tmp_path / f"song{i}.wav"
        sf.write(str(p), _voiced(sr, 5.0, freq=200 + 20 * i), sr)
        song_paths.append(str(p))

    out_root = tmp_path / "dataset"
    summary = build_singer_dataset(
        "rafi", song_paths, str(out_root), cfg, separate=False)

    assert summary["singer"] == "rafi"
    assert summary["inputs_processed"] == 2
    assert summary["inputs_failed"] == 0
    assert summary["num_segments"] > 0
    assert os.path.exists(summary["manifest"])

    rows = read_manifest(summary["manifest"])
    assert len(rows) == summary["num_segments"]
    assert all(os.path.exists(r["path"]) for r in rows)


def test_build_singer_dataset_handles_bad_input(cfg, tmp_path):
    out_root = tmp_path / "dataset"
    summary = build_singer_dataset(
        "rafi", ["/nonexistent.wav"], str(out_root), cfg, separate=False)
    assert summary["inputs_failed"] == 1
    assert summary["num_segments"] == 0
