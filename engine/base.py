"""
Core engine abstractions: a target-voice descriptor and the conversion
interface that every backend must implement.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class VoiceProfile:
    """Describes a target singer voice.

    Attributes:
        id: Stable identifier (used in API requests).
        name: Human-readable display name.
        reference_audio: Path to a clean reference clip of the target voice
            (used by embedding/reference-based engines such as AutoVC).
        licensed: Whether we hold rights to this voice. The service refuses to
            convert to voices that are not licensed, which keeps the rights
            policy enforced in code rather than in a wiki.
        description: Optional free-text description.
        metadata: Arbitrary extra fields (model path, embedding path, …).
    """
    id: str
    name: str
    reference_audio: Optional[str] = None
    licensed: bool = False
    description: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_public_dict(self) -> Dict:
        """Serializable view safe to expose via the API."""
        return {
            "id": self.id,
            "name": self.name,
            "licensed": self.licensed,
            "description": self.description,
        }


class ConversionEngine(ABC):
    """Interface every singing-voice-conversion backend implements."""

    #: Short backend identifier, e.g. "autovc", "rvc", "so-vits-svc".
    name: str = "base"

    def load(self) -> None:
        """Lazily initialize heavy resources (models, weights). Optional."""

    @abstractmethod
    def convert(
        self,
        source_audio: np.ndarray,
        sr: int,
        voice: VoiceProfile,
    ) -> Tuple[np.ndarray, int]:
        """Convert source vocals toward the target voice.

        Args:
            source_audio: Mono source vocals (float32).
            sr: Sample rate of ``source_audio``.
            voice: Target voice profile.

        Returns:
            (converted_audio, sample_rate)
        """
        raise NotImplementedError
