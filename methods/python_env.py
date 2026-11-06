from __future__ import annotations

import os
import venv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"


def python_executable() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_venv() -> Path:
    python = python_executable()
    if python.is_file():
        return python
    if VENV_DIR.exists():
        raise RuntimeError(f"Incomplete shared Python environment: {VENV_DIR}")
    venv.create(VENV_DIR, with_pip=True)
    if not python.is_file():
        raise FileNotFoundError(f"Failed to create shared Python environment: {VENV_DIR}")
    return python
