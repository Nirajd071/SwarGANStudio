"""
Configuration for the singer data pipeline.

These defaults favour dataset *fidelity* (44.1 kHz vocals) so the produced
dataset is model-agnostic; individual training stacks (RVC, so-vits-svc, …)
can downsample as needed.
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class PipelineConfig:
    # --- Audio format -------------------------------------------------------
    sample_rate: int = 44100          # dataset master sample rate
    mono: bool = True

    # --- Cleaning -----------------------------------------------------------
    highpass_hz: float = 40.0         # remove sub-bass rumble / DC
    peak_target: float = 0.95         # peak-normalize target amplitude
    trim_top_db: float = 30.0         # silence threshold for edge trimming

    # --- Segmentation -------------------------------------------------------
    segment_seconds: float = 6.0      # target segment length
    min_segment_seconds: float = 2.0  # discard shorter than this
    max_segment_seconds: float = 12.0 # split anything longer
    split_top_db: float = 30.0        # silence threshold for splitting

    # --- Quality filtering --------------------------------------------------
    min_rms: float = 0.01             # discard near-silent segments
    min_voiced_ratio: float = 0.35    # discard segments with too little pitch
    clip_peak: float = 0.999          # warn/flag segments that clip
    f0_min_hz: float = 65.0           # plausible singing pitch range
    f0_max_hz: float = 1200.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
