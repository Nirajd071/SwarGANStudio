"""
Dataset manifest: write accepted segments to disk and record one JSONL line
of metadata per segment, plus a dataset-level summary.
"""
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf

from data_pipeline.config import PipelineConfig
from data_pipeline.segment import SegmentQuality
from utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ManifestEntry:
    id: str
    singer: str
    source: str
    path: str
    sample_rate: int
    duration: float
    rms: float
    peak: float
    voiced_ratio: float
    f0_median_hz: float
    f0_mean_hz: float

    def to_dict(self) -> Dict:
        return asdict(self)


def write_segments(
    accepted: List[Tuple[np.ndarray, SegmentQuality]],
    singer: str,
    source: str,
    output_dir: str,
    cfg: PipelineConfig,
) -> List[ManifestEntry]:
    """Write each accepted segment to ``output_dir`` and build manifest entries."""
    os.makedirs(output_dir, exist_ok=True)
    source_stem = Path(source).stem
    entries: List[ManifestEntry] = []

    for idx, (seg, quality) in enumerate(accepted):
        seg_id = f"{singer}_{source_stem}_{idx:04d}"
        out_path = os.path.join(output_dir, f"{seg_id}.wav")
        sf.write(out_path, seg.astype(np.float32), cfg.sample_rate)
        entries.append(ManifestEntry(
            id=seg_id,
            singer=singer,
            source=source,
            path=out_path,
            sample_rate=cfg.sample_rate,
            duration=round(quality.duration, 4),
            rms=round(quality.rms, 6),
            peak=round(quality.peak, 6),
            voiced_ratio=round(quality.voiced_ratio, 4),
            f0_median_hz=round(quality.f0_median_hz, 2),
            f0_mean_hz=round(quality.f0_mean_hz, 2),
        ))

    return entries


def write_manifest(entries: List[ManifestEntry], path: str) -> None:
    """Write manifest entries as JSONL."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict()) + "\n")
    logger.info("Wrote manifest with %d entries -> %s", len(entries), path)


def read_manifest(path: str) -> List[Dict]:
    """Read a JSONL manifest into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(entries: List[ManifestEntry]) -> Dict:
    """Compute dataset-level summary statistics."""
    if not entries:
        return {"num_segments": 0, "total_duration_s": 0.0, "singers": {}}

    durations = [e.duration for e in entries]
    by_singer: Dict[str, Dict] = {}
    for e in entries:
        s = by_singer.setdefault(e.singer, {"num_segments": 0, "duration_s": 0.0})
        s["num_segments"] += 1
        s["duration_s"] = round(s["duration_s"] + e.duration, 3)

    return {
        "num_segments": len(entries),
        "total_duration_s": round(float(np.sum(durations)), 3),
        "mean_segment_s": round(float(np.mean(durations)), 3),
        "singers": by_singer,
    }


def write_summary(entries: List[ManifestEntry], path: str) -> Dict:
    """Write the dataset summary as JSON and return it."""
    summary = summarize(entries)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Dataset summary: %d segments, %.1fs total across %d singer(s)",
                summary["num_segments"], summary["total_duration_s"],
                len(summary["singers"]))
    return summary
