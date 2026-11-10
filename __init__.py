"""Knowing versus Naming -- memory as a bias channel in LLM scholar recommendation."""

from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> None:
    """Read `.env` at the repo root into os.environ.

    Written out rather than pulling in python-dotenv: it is twelve lines, and
    it keeps the dependency list short enough that the whole project installs
    on a cluster login node without a wheel build.

    Values already present in the real environment win, so an exported
    OPENALEX_API_KEY overrides the file, which is what you want on a shared
    machine where the file may hold someone else's key.
    """
    env_path = path or ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


load_env()
