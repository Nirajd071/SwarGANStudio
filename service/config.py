"""Service configuration."""
import os
import tempfile
from dataclasses import dataclass, field


def _default_storage() -> str:
    return os.environ.get(
        "SWARGAN_STORAGE_DIR",
        os.path.join(tempfile.gettempdir(), "swargan_storage"))


@dataclass
class ServiceSettings:
    storage_dir: str = field(default_factory=_default_storage)
    max_workers: int = 2
    # Reject uploads larger than this (defends against memory blowups).
    max_upload_mb: int = 50
    # Allow conversion to voices we do not hold rights to (default: no).
    allow_unlicensed: bool = False
