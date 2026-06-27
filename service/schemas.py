"""Pydantic response models for the API."""
from typing import List, Optional

from pydantic import BaseModel


class VoiceOut(BaseModel):
    id: str
    name: str
    licensed: bool
    description: str = ""


class VoiceListOut(BaseModel):
    voices: List[VoiceOut]


class JobOut(BaseModel):
    id: str
    status: str
    voice_id: str
    engine: str
    separate: bool
    error: Optional[str] = None
    result_available: bool = False


class HealthOut(BaseModel):
    status: str
    engines: List[str]
    num_voices: int
