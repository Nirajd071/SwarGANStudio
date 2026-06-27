"""
Ingestion stage: optional download (yt-dlp) and vocal separation (Demucs).

These wrap external tools. They import yt-dlp lazily and check for the
presence of ffmpeg/demucs so the rest of the pipeline can be used (and
tested) without them installed.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from utils.logging_config import get_logger

logger = get_logger(__name__)


def ffmpeg_available() -> bool:
    """Return True if an ffmpeg binary is on PATH."""
    return shutil.which("ffmpeg") is not None


def demucs_available() -> bool:
    """Return True if the demucs package is importable."""
    try:
        import demucs  # noqa: F401
        return True
    except Exception:
        return False


def download_audio(url: str, output_path: str) -> str:
    """Download the best audio track from a URL to ``output_path`` (mp3).

    Args:
        url: Source URL (YouTube, SoundCloud, …).
        output_path: Destination path *without* extension.

    Returns:
        Path to the downloaded ``.mp3`` file.

    Raises:
        RuntimeError: If ffmpeg or yt-dlp are unavailable, or download fails.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is required for downloading but was not found on PATH.")
    try:
        import yt_dlp
    except Exception as e:  # pragma: no cover - depends on optional dep
        raise RuntimeError(f"yt-dlp is required for downloading: {e}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path + ".%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "noplaylist": True,
        "quiet": True,
    }

    logger.info("Downloading audio from %s", url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    final_path = output_path + ".mp3"
    if not os.path.exists(final_path):
        # Fall back to whatever extension landed on disk
        for ext in (".m4a", ".webm", ".opus", ".wav"):
            candidate = output_path + ext
            if os.path.exists(candidate):
                return candidate
        raise RuntimeError("Download completed but no output file was found.")
    return final_path


def separate_vocals(
    input_path: str,
    output_dir: str,
    model_name: str = "htdemucs",
    timeout: int = 1800,
) -> str:
    """Separate vocals from a track using Demucs (CLI).

    Args:
        input_path: Path to the input audio file.
        output_dir: Directory where Demucs writes its stems.
        model_name: Demucs model to use.
        timeout: Subprocess timeout in seconds.

    Returns:
        Path to the extracted ``vocals.wav``.

    Raises:
        RuntimeError: If demucs is unavailable or separation fails.
    """
    if not demucs_available():
        raise RuntimeError("demucs is not installed; cannot separate vocals.")

    os.makedirs(output_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "demucs", "-n", model_name,
           "--two-stems", "vocals", "-o", output_dir, input_path]

    logger.info("Separating vocals with Demucs (%s)", model_name)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr.strip()[-500:]}")

    stem = Path(input_path).stem
    vocals_path = os.path.join(output_dir, model_name, stem, "vocals.wav")
    if not os.path.exists(vocals_path):
        raise RuntimeError(f"Expected vocals stem not found at {vocals_path}")
    return vocals_path
