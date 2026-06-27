"""FastAPI service exposing the SwarGAN conversion pipeline as an async API."""
from service.app import create_app

__all__ = ["create_app"]
