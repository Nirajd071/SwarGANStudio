"""
Command-line interface for the singer data pipeline.

Examples
--------
Build a dataset from already-separated vocal stems::

    python -m data_pipeline.cli --singer rafi --output datasets stem1.wav stem2.wav

Build from full tracks, running Demucs separation first::

    python -m data_pipeline.cli --singer rafi --output datasets --separate song1.mp3

Build from URLs (requires ffmpeg + yt-dlp + demucs)::

    python -m data_pipeline.cli --singer rafi --output datasets --urls https://... https://...
"""
import argparse
import json
import sys
from typing import List, Optional

from data_pipeline.config import PipelineConfig
from data_pipeline.pipeline import build_singer_dataset, build_from_urls
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="data_pipeline",
        description="Build a cleaned, segmented vocal dataset for a singer.")
    p.add_argument("--singer", required=True, help="Singer identifier.")
    p.add_argument("--output", required=True, help="Output root directory.")
    p.add_argument("inputs", nargs="*", help="Input audio files (tracks or vocal stems).")
    p.add_argument("--urls", nargs="*", default=None,
                   help="Download these URLs first (implies separation).")
    p.add_argument("--separate", action="store_true",
                   help="Run Demucs vocal separation on local inputs first.")
    p.add_argument("--sample-rate", type=int, default=None,
                   help="Override dataset sample rate.")
    p.add_argument("--segment-seconds", type=float, default=None,
                   help="Override target segment length (seconds).")
    p.add_argument("--min-voiced-ratio", type=float, default=None,
                   help="Override minimum voiced ratio for keeping a segment.")
    return p


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    if args.sample_rate is not None:
        cfg.sample_rate = args.sample_rate
    if args.segment_seconds is not None:
        cfg.segment_seconds = args.segment_seconds
    if args.min_voiced_ratio is not None:
        cfg.min_voiced_ratio = args.min_voiced_ratio
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _config_from_args(args)

    if not args.urls and not args.inputs:
        logger.error("Provide input files or --urls.")
        return 2

    if args.urls:
        summary = build_from_urls(args.singer, args.urls, args.output, cfg)
    else:
        summary = build_singer_dataset(
            args.singer, args.inputs, args.output, cfg, separate=args.separate)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
