"""
Conversion engine abstraction.

This package decouples the *product* (separation -> conversion -> remix) from
the *model*. A `ConversionEngine` is a pluggable backend: today the reference
implementation wraps the in-repo AutoVC model, but an RVC / so-vits-svc /
diffusion backend can be dropped in later without touching the service layer.
"""
from engine.base import ConversionEngine, VoiceProfile
from engine.registry import VoiceRegistry, EngineRegistry, default_registries
from engine.external_engine import ExternalCommandEngine

__all__ = [
    "ConversionEngine",
    "VoiceProfile",
    "VoiceRegistry",
    "EngineRegistry",
    "default_registries",
    "ExternalCommandEngine",
]
