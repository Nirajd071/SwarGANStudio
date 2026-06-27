"""
FastAPI application factory for the SwarGAN conversion service.

Endpoints
---------
GET  /health               liveness + capability summary
GET  /voices               list the offered (licensed) target voices
POST /convert              submit a conversion job (multipart upload)
GET  /jobs/{job_id}        poll job status
GET  /jobs/{job_id}/result download the converted audio when ready
"""
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from engine.registry import VoiceRegistry, EngineRegistry, default_registries
from service.config import ServiceSettings
from service.conversion import run_conversion
from service.jobs import JobManager, JobStatus
from service.schemas import VoiceOut, VoiceListOut, JobOut, HealthOut
from service.storage import Storage
from utils.logging_config import get_logger

logger = get_logger(__name__)


def create_app(
    voices: Optional[VoiceRegistry] = None,
    engines: Optional[EngineRegistry] = None,
    settings: Optional[ServiceSettings] = None,
    job_manager: Optional[JobManager] = None,
) -> FastAPI:
    """Build the FastAPI app. Dependencies are injectable for testing."""
    if voices is None or engines is None:
        default_voices, default_engines = default_registries()
        voices = voices or default_voices
        engines = engines or default_engines
    settings = settings or ServiceSettings()
    storage = Storage(settings)
    jobs = job_manager or JobManager(max_workers=settings.max_workers)

    app = FastAPI(title="SwarGAN Studio API", version="0.1.0")

    @app.get("/health", response_model=HealthOut)
    def health():
        return HealthOut(status="ok", engines=engines.names(),
                         num_voices=len(voices.list()))

    @app.get("/voices", response_model=VoiceListOut)
    def list_voices():
        # Only advertise voices we are allowed to offer.
        offered = voices.list() if settings.allow_unlicensed else voices.list_licensed()
        return VoiceListOut(voices=[VoiceOut(**v.to_public_dict()) for v in offered])

    def _resolve_voice_and_engine(voice_id: str, engine: Optional[str]):
        """Validate the target voice (incl. licensing) and the engine."""
        voice = voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=404, detail=f"Unknown voice: {voice_id}")
        if not voice.licensed and not settings.allow_unlicensed:
            raise HTTPException(
                status_code=403,
                detail=("Conversion to this voice is not permitted: we do not "
                        "hold a license for it."))
        try:
            eng = engines.get(engine)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        return voice, eng

    async def _read_and_store_upload(file: UploadFile) -> str:
        """Validate upload size/emptiness and persist it; return the path."""
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty upload.")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {settings.max_upload_mb} MB limit.")
        suffix = os.path.splitext(file.filename or "")[1] or ".wav"
        return storage.save_upload(data, suffix=suffix)

    @app.post("/convert", response_model=JobOut, status_code=202)
    async def convert(
        file: UploadFile = File(...),
        voice_id: str = Form(...),
        separate: bool = Form(False),
        engine: Optional[str] = Form(None),
    ):
        """Submit an asynchronous conversion job (poll /jobs/{id})."""
        voice, eng = _resolve_voice_and_engine(voice_id, engine)
        input_path = await _read_and_store_upload(file)

        job = jobs.create(voice_id=voice_id, engine=eng.name, separate=separate)
        output_path = storage.result_path(job.id)

        def _task() -> str:
            return run_conversion(input_path, voice, eng, output_path,
                                  separate=separate)

        jobs.submit(job, _task)
        return JobOut(**job.to_dict())

    @app.post("/convert/sync")
    async def convert_sync(
        file: UploadFile = File(...),
        voice_id: str = Form(...),
        separate: bool = Form(False),
        engine: Optional[str] = Form(None),
    ):
        """Convert synchronously and return the audio directly.

        Convenient for short clips / simple clients that prefer a single
        blocking call over the submit-and-poll job flow. Heavy/long jobs
        should still use the asynchronous /convert endpoint.
        """
        voice, eng = _resolve_voice_and_engine(voice_id, engine)
        input_path = await _read_and_store_upload(file)

        import uuid
        result_id = uuid.uuid4().hex
        output_path = storage.result_path(result_id)
        try:
            # Conversion is CPU-bound; run it off the event loop.
            await run_in_threadpool(
                run_conversion, input_path, voice, eng, output_path, separate)
        except Exception as e:  # noqa: BLE001
            logger.exception("Synchronous conversion failed.")
            raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

        return FileResponse(output_path, media_type="audio/wav",
                            filename=f"converted_{result_id}.wav")

    @app.get("/jobs/{job_id}", response_model=JobOut)
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        return JobOut(**job.to_dict())

    @app.get("/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job.")
        if job.status == JobStatus.FAILED:
            raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")
        if job.status != JobStatus.DONE or not job.result_path:
            raise HTTPException(status_code=409, detail=f"Job not ready: {job.status}")
        return FileResponse(job.result_path, media_type="audio/wav",
                            filename=f"converted_{job_id}.wav")

    app.state.voices = voices
    app.state.engines = engines
    app.state.jobs = jobs
    app.state.storage = storage
    return app
