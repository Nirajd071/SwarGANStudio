"""
Centralized logging configuration for SwarGAN.

Use ``get_logger(__name__)`` in any module to obtain a configured logger.
The log level can be controlled with the ``SWARGAN_LOG_LEVEL`` environment
variable (e.g. DEBUG, INFO, WARNING). Defaults to INFO.
"""
import logging
import os

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _configure_root() -> None:
    """Configure the root logger once for the whole process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("SWARGAN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))

    root = logging.getLogger("swargan")
    root.setLevel(level)
    # Avoid duplicate handlers if reconfigured
    if not root.handlers:
        root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under 'swargan'.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured ``logging.Logger`` instance.
    """
    _configure_root()
    # Namespace all app loggers under "swargan" so they share configuration.
    short = name.split(".")[-1]
    return logging.getLogger(f"swargan.{short}")
