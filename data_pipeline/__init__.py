"""
SwarGAN Studio — singer data pipeline.

A model-agnostic pipeline that turns raw songs into clean, segmented,
quality-filtered vocal datasets ready for singing-voice-conversion training:

    download (yt-dlp)  ->  separate (Demucs)  ->  clean  ->  segment  ->  manifest

The cleaning, segmentation and manifest stages depend only on numpy/librosa/
scipy/pyworld/soundfile and are fully unit-tested. The download and
separation stages wrap external tools (ffmpeg, demucs) and degrade gracefully
when those tools are not installed.
"""
from data_pipeline.config import PipelineConfig

__all__ = ["PipelineConfig"]
