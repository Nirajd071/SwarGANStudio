"""
In-process async job manager.

A thread-pool backed job store suitable for a single-node deployment / demo.
The interface (create / submit / get) is intentionally small so it can later
be swapped for a real queue (Celery, RQ, SQS + workers) without touching the
API layer.
"""
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    voice_id: str
    engine: str
    separate: bool
    status: str = JobStatus.PENDING
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "status": self.status,
            "voice_id": self.voice_id,
            "engine": self.engine,
            "separate": self.separate,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_available": self.status == JobStatus.DONE,
        }


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, voice_id: str, engine: str, separate: bool) -> Job:
        job = Job(id=uuid.uuid4().hex, voice_id=voice_id,
                  engine=engine, separate=separate)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def submit(self, job: Job, fn: Callable[[], str]) -> None:
        """Run ``fn`` in the background; ``fn`` must return the result path."""
        def _run():
            self._update(job, status=JobStatus.RUNNING)
            try:
                result_path = fn()
                self._update(job, status=JobStatus.DONE, result_path=result_path)
            except Exception as e:  # noqa: BLE001 - surface any failure to the job
                self._update(job, status=JobStatus.FAILED, error=str(e))

        self._executor.submit(_run)

    def _update(self, job: Job, **changes) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(job, key, value)
            job.updated_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
