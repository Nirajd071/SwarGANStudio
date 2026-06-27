"""
Ready-made command templates for popular open-source singing-voice-conversion
backends, wrapped as ``ExternalCommandEngine`` instances.

These let you flip SwarGAN's conversion to a real, high-quality backend once you
have the corresponding tool installed and a trained model. The factories bake
your model/config paths in as literals, leaving only ``{source}`` and
``{output}`` as runtime placeholders that the engine fills per request.

Exact CLI flags vary between forks/versions — treat these as sensible defaults
and adjust the templates to match your installation (see docs/SVC_BACKENDS.md).
"""
import sys
from typing import List, Optional

from engine.external_engine import ExternalCommandEngine


def make_so_vits_svc_engine(
    model_path: str,
    config_path: str,
    speaker: str,
    svc_command: str = "svc",
    extra_args: Optional[List[str]] = None,
    timeout: int = 1800,
) -> ExternalCommandEngine:
    """Build an engine for so-vits-svc (e.g. the ``so-vits-svc-fork`` CLI).

    Mirrors: ``svc infer INPUT -m MODEL -c CONFIG -s SPEAKER -o OUTPUT``
    """
    template = [
        svc_command, "infer", "{source}",
        "-m", model_path,
        "-c", config_path,
        "-s", speaker,
        "-o", "{output}",
    ]
    if extra_args:
        template.extend(extra_args)
    return ExternalCommandEngine(
        command_template=template, name="so-vits-svc", timeout=timeout)


def make_rvc_engine(
    model_path: str,
    index_path: Optional[str] = None,
    f0_method: str = "rmvpe",
    transpose: int = 0,
    rvc_command: str = "rvc",
    extra_args: Optional[List[str]] = None,
    timeout: int = 1800,
) -> ExternalCommandEngine:
    """Build an engine for RVC (e.g. the ``rvc-python`` CLI).

    Mirrors: ``rvc infer -i INPUT -o OUTPUT -mp MODEL [-ip INDEX]
                  -me F0METHOD -pi TRANSPOSE``
    """
    template = [
        rvc_command, "infer",
        "-i", "{source}",
        "-o", "{output}",
        "-mp", model_path,
        "-me", f0_method,
        "-pi", str(transpose),
    ]
    if index_path:
        template.extend(["-ip", index_path])
    if extra_args:
        template.extend(extra_args)
    return ExternalCommandEngine(
        command_template=template, name="rvc", timeout=timeout)


def make_python_script_engine(
    script_path: str,
    name: str = "custom-svc",
    python_executable: Optional[str] = None,
    extra_args: Optional[List[str]] = None,
    timeout: int = 1800,
) -> ExternalCommandEngine:
    """Generic helper for a ``python infer.py --input {source} --output {output}``
    style script. ``extra_args`` may reference any of the engine placeholders
    ({source}, {output}, {target}, {voice_id}, {model}, or voice metadata)."""
    template = [
        python_executable or sys.executable, script_path,
        "--input", "{source}",
        "--output", "{output}",
    ]
    if extra_args:
        template.extend(extra_args)
    return ExternalCommandEngine(
        command_template=template, name=name, timeout=timeout)
