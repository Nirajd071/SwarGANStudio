"""
Registries for voices and engines.

- VoiceRegistry: the catalogue of target voices the platform offers.
- EngineRegistry: the available conversion backends.
"""
import os
from typing import Dict, List, Optional

from engine.base import ConversionEngine, VoiceProfile
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VoiceRegistry:
    """In-memory catalogue of target voices."""

    def __init__(self):
        self._voices: Dict[str, VoiceProfile] = {}

    def register(self, voice: VoiceProfile) -> None:
        self._voices[voice.id] = voice

    def get(self, voice_id: str) -> Optional[VoiceProfile]:
        return self._voices.get(voice_id)

    def list(self) -> List[VoiceProfile]:
        return list(self._voices.values())

    def list_licensed(self) -> List[VoiceProfile]:
        return [v for v in self._voices.values() if v.licensed]

    def load_from_directory(self, directory: str, licensed: bool = False) -> int:
        """Register one voice per audio file found in ``directory``.

        Filenames become voice ids/names. Intended for demo/original voices.
        Returns the number of voices registered.
        """
        if not directory or not os.path.isdir(directory):
            return 0
        count = 0
        for fname in sorted(os.listdir(directory)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in (".wav", ".flac", ".mp3", ".ogg"):
                continue
            self.register(VoiceProfile(
                id=stem,
                name=stem.replace("_", " ").title(),
                reference_audio=os.path.join(directory, fname),
                licensed=licensed,
                description=f"Voice loaded from {fname}",
            ))
            count += 1
        logger.info("Loaded %d voice(s) from %s", count, directory)
        return count


class EngineRegistry:
    """Catalogue of conversion engines keyed by name."""

    def __init__(self):
        self._engines: Dict[str, ConversionEngine] = {}
        self._default: Optional[str] = None

    def register(self, engine: ConversionEngine, default: bool = False) -> None:
        self._engines[engine.name] = engine
        if default or self._default is None:
            self._default = engine.name

    def get(self, name: Optional[str] = None) -> ConversionEngine:
        key = name or self._default
        if key is None or key not in self._engines:
            raise KeyError(f"Unknown engine: {name!r}")
        return self._engines[key]

    def names(self) -> List[str]:
        return list(self._engines.keys())


def default_registries() -> (VoiceRegistry, EngineRegistry):
    """Build registries with the reference AutoVC engine registered.

    No voices are registered by default — the operator supplies licensed /
    original voices (e.g. via ``VoiceRegistry.load_from_directory``).
    """
    from engine.autovc_engine import AutoVCEngine

    voices = VoiceRegistry()
    engines = EngineRegistry()
    engines.register(AutoVCEngine(), default=True)
    return voices, engines
