"""
Generic external conversion engine.

Wraps any command-line SVC tool (RVC, so-vits-svc, DDSP-SVC, …) behind the
``ConversionEngine`` interface. The command is described as a template whose
placeholders are filled at call time:

    {source}   path to the input vocals wav (written by this engine)
    {output}   path the tool must write the converted wav to
    {target}   the voice's reference audio path (may be empty)
    {voice_id} the voice id
    {model}    the voice's model path from metadata['model'] (may be empty)

Example (illustrative) so-vits-svc-style template::

    [sys.executable, "infer.py",
     "--model", "{model}", "--input", "{source}", "--output", "{output}"]

This keeps the heavy/GPU tool out-of-process and swappable without touching
the service layer.
"""
import os
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np
import soundfile as sf

from engine.base import ConversionEngine, VoiceProfile
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ExternalCommandEngine(ConversionEngine):
    def __init__(
        self,
        command_template: List[str],
        name: str = "external",
        timeout: int = 1800,
        output_format: str = "wav",
    ):
        if not command_template:
            raise ValueError("command_template must be a non-empty list.")
        self.name = name
        self.command_template = command_template
        self.timeout = timeout
        self.output_format = output_format

    def _render(self, mapping: dict) -> List[str]:
        return [part.format(**mapping) for part in self.command_template]

    def convert(
        self,
        source_audio: np.ndarray,
        sr: int,
        voice: VoiceProfile,
    ) -> Tuple[np.ndarray, int]:
        source_audio = np.ascontiguousarray(
            np.asarray(source_audio, dtype=np.float32).reshape(-1))

        with tempfile.TemporaryDirectory(prefix="swargan_ext_") as tmp:
            source_path = os.path.join(tmp, "source.wav")
            output_path = os.path.join(tmp, f"output.{self.output_format}")
            sf.write(source_path, source_audio, sr)

            mapping = {
                "source": source_path,
                "output": output_path,
                "target": voice.reference_audio or "",
                "voice_id": voice.id,
                "model": voice.metadata.get("model", ""),
            }
            # Allow templates to reference arbitrary voice metadata, e.g.
            # {config}, {speaker}, {index} for RVC / so-vits-svc backends.
            mapping.update({k: str(v) for k, v in voice.metadata.items()})
            cmd = self._render(mapping)
            logger.info("Running external engine '%s': %s", self.name, " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"External engine '{self.name}' timed out after {self.timeout}s.")

            if result.returncode != 0:
                raise RuntimeError(
                    f"External engine '{self.name}' failed "
                    f"(exit {result.returncode}): {result.stderr.strip()[-500:]}")

            if not os.path.exists(output_path):
                raise RuntimeError(
                    f"External engine '{self.name}' produced no output file.")

            audio, out_sr = sf.read(output_path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return np.ascontiguousarray(audio, dtype=np.float32), int(out_sr)
