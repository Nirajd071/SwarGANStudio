"""Filesystem storage for uploads and conversion results."""
import os
import uuid

from service.config import ServiceSettings


class Storage:
    def __init__(self, settings: ServiceSettings):
        self.settings = settings
        self.uploads_dir = os.path.join(settings.storage_dir, "uploads")
        self.results_dir = os.path.join(settings.storage_dir, "results")
        os.makedirs(self.uploads_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def save_upload(self, data: bytes, suffix: str = ".wav") -> str:
        name = f"{uuid.uuid4().hex}{suffix}"
        path = os.path.join(self.uploads_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def result_path(self, job_id: str) -> str:
        return os.path.join(self.results_dir, f"{job_id}.wav")
