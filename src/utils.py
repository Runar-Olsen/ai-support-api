# src/utils.py
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_data_path() -> Path:
    p = PROJECT_ROOT / "data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def load_env() -> None:
    """Load variables from .env file into environment."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set.")
    return value


def is_mock_mode() -> bool:
    """Returns True if OPENAI_API_KEY is missing, meaning mock mode is active."""
    load_env()
    return os.getenv("OPENAI_API_KEY") is None
