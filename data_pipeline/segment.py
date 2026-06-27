"""
Segmentation + quality filtering.

Splits a cleaned vocal track into training-sized segments and keeps only
those that look like actual sung vocals (rejecting silence, instrumental
bleed and too-short fragments).
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import librosa
import pyworld as pw

from data_pipeline.config import PipelineConfig
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SegmentQuality:
    duration: float
    rms: float
    peak: float
    voiced_ratio: float
    f0_median_hz: float
    f0_mean_hz: float

    def to_dict(self):
        return {
            "duration": round(self.duration, 4),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "voiced_ratio": round(self.voiced_ratio, 4),
            "f0_median_hz": round(self.f0_median_hz, 2),
            "f0_mean_hz": round(self.f0_mean_hz, 2),
        }


def compute_quality(audio: np.ndarray, sr: int) -> SegmentQuality:
    """Compute energy + pitch statistics used for quality filtering."""
    audio = np.ascontiguousarray(audio, dtype=np.float64)
    n = audio.shape[0]
    duration = n / sr
    rms = float(np.sqrt(np.mean(audio ** 2))) if n else 0.0
    peak = float(np.max(np.abs(audio))) if n else 0.0

    # WORLD DIO for voicing / pitch statistics
    f0, time_axis = pw.dio(audio, sr, frame_period=10.0)
    f0 = pw.stonemask(audio, f0, time_axis, sr)
    voiced = f0 > 0
    voiced_ratio = float(np.mean(voiced)) if f0.size else 0.0
    if np.any(voiced):
        f0_voiced = f0[voiced]
        f0_median = float(np.median(f0_voiced))
        f0_mean = float(np.mean(f0_voiced))
    else:
        f0_median = 0.0
        f0_mean = 0.0

    return SegmentQuality(duration, rms, peak, voiced_ratio, f0_median, f0_mean)


def _chunk_interval(seg: np.ndarray, sr: int, cfg: PipelineConfig) -> List[np.ndarray]:
    """Split a non-silent interval into target-length chunks."""
    target = int(cfg.segment_seconds * sr)
    min_len = int(cfg.min_segment_seconds * sr)
    max_len = int(cfg.max_segment_seconds * sr)
    n = seg.shape[0]

    if n < min_len:
        return []
    if n <= max_len:
        return [seg]

    chunks = []
    start = 0
    while start < n:
        end = min(start + target, n)
        chunk = seg[start:end]
        if chunk.shape[0] >= min_len:
            chunks.append(chunk)
        start = end
    return chunks


def segment_audio(audio: np.ndarray, sr: int, cfg: PipelineConfig) -> List[np.ndarray]:
    """Split a track into candidate segments on silence boundaries."""
    if audio.size == 0:
        return []
    intervals = librosa.effects.split(audio, top_db=cfg.split_top_db)
    segments: List[np.ndarray] = []
    for start, end in intervals:
        segments.extend(_chunk_interval(audio[start:end], sr, cfg))
    return segments


def accept_segment(quality: SegmentQuality, cfg: PipelineConfig) -> Tuple[bool, str]:
    """Decide whether a segment is good enough to keep.

    Returns (accepted, reason). ``reason`` explains a rejection or is "ok".
    """
    if quality.duration < cfg.min_segment_seconds:
        return False, "too_short"
    if quality.rms < cfg.min_rms:
        return False, "too_quiet"
    if quality.voiced_ratio < cfg.min_voiced_ratio:
        # Low pitch presence usually means instrumental bleed or noise
        return False, "low_voiced_ratio"
    if quality.f0_median_hz and not (
            cfg.f0_min_hz <= quality.f0_median_hz <= cfg.f0_max_hz):
        return False, "pitch_out_of_range"
    return True, "ok"


def segment_and_filter(
    audio: np.ndarray, sr: int, cfg: PipelineConfig
) -> Tuple[List[Tuple[np.ndarray, SegmentQuality]], dict]:
    """Segment a track and keep only acceptable segments.

    Returns:
        (accepted, reject_counts) where ``accepted`` is a list of
        (segment_audio, quality) and ``reject_counts`` maps reason -> count.
    """
    accepted: List[Tuple[np.ndarray, SegmentQuality]] = []
    reject_counts: dict = {}

    for seg in segment_audio(audio, sr, cfg):
        quality = compute_quality(seg, sr)
        ok, reason = accept_segment(quality, cfg)
        if ok:
            accepted.append((seg, quality))
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    logger.info("Segmentation kept %d segments (rejected: %s)",
                len(accepted), reject_counts or "none")
    return accepted, reject_counts
