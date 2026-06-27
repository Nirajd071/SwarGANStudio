"""
Pipeline orchestrator: tie ingestion, cleaning, segmentation and manifest
writing into a single per-singer dataset build.
"""
import os
import tempfile
from typing import Dict, List, Optional

from data_pipeline.config import PipelineConfig
from data_pipeline.clean import clean_vocal
from data_pipeline.segment import segment_and_filter
from data_pipeline.manifest import write_segments, write_manifest, write_summary
from data_pipeline.ingest import download_audio, separate_vocals
from utils.logging_config import get_logger

logger = get_logger(__name__)


def build_singer_dataset(
    singer: str,
    input_paths: List[str],
    output_root: str,
    cfg: Optional[PipelineConfig] = None,
    separate: bool = False,
) -> Dict:
    """Build a cleaned, segmented dataset for a single singer.

    Args:
        singer: Singer identifier (used in segment ids and output dir).
        input_paths: Local audio files. These may be full tracks (set
            ``separate=True`` to run Demucs first) or already-isolated vocals.
        output_root: Root directory for outputs.
        cfg: Pipeline configuration (defaults to ``PipelineConfig()``).
        separate: If True, run Demucs vocal separation on each input first.

    Returns:
        Dataset summary dict (also written to ``<singer>.summary.json``).
    """
    cfg = cfg or PipelineConfig()
    seg_dir = os.path.join(output_root, singer)
    stems_dir = os.path.join(output_root, "_stems")

    all_entries = []
    reject_totals: Dict[str, int] = {}
    processed, failed = 0, 0

    for path in input_paths:
        try:
            vocal_path = separate_vocals(path, stems_dir) if separate else path

            audio = clean_vocal(vocal_path, cfg)
            accepted, rejects = segment_and_filter(audio, cfg.sample_rate, cfg)
            entries = write_segments(accepted, singer, path, seg_dir, cfg)
            all_entries.extend(entries)
            for reason, count in rejects.items():
                reject_totals[reason] = reject_totals.get(reason, 0) + count
            processed += 1
        except Exception as e:
            failed += 1
            logger.error("Failed to process %s: %s", path, e)

    manifest_path = os.path.join(output_root, f"{singer}.manifest.jsonl")
    summary_path = os.path.join(output_root, f"{singer}.summary.json")
    write_manifest(all_entries, manifest_path)
    summary = write_summary(all_entries, summary_path)
    summary.update({
        "singer": singer,
        "inputs_processed": processed,
        "inputs_failed": failed,
        "rejected": reject_totals,
        "manifest": manifest_path,
    })
    return summary


def build_from_urls(
    singer: str,
    urls: List[str],
    output_root: str,
    cfg: Optional[PipelineConfig] = None,
) -> Dict:
    """Download tracks from URLs, then build a dataset (download -> separate -> …).

    Requires ffmpeg, yt-dlp and demucs to be installed.
    """
    cfg = cfg or PipelineConfig()
    downloaded: List[str] = []
    with tempfile.TemporaryDirectory(prefix="swargan_dl_") as tmp:
        for i, url in enumerate(urls):
            try:
                path = download_audio(url, os.path.join(tmp, f"track_{i:03d}"))
                downloaded.append(path)
            except Exception as e:
                logger.error("Download failed for %s: %s", url, e)
        return build_singer_dataset(singer, downloaded, output_root, cfg, separate=True)
