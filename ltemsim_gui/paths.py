"""Filesystem locations used by the GUI.

The GUI never writes into the hand-curated result folders by default: every run it
launches goes to results/gui_runs/<kind>_<timestamp>/ unless the user picks a folder.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # repository root
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "results"
DATA = ROOT / "data"
DOCS = ROOT / "docs"
GUI_RUNS = RESULTS / "gui_runs"
MEASURED_CSV = DATA / "measured_tip_scan.csv"
ASSETS = Path(__file__).resolve().parent / "assets"

APP_NAME = "LTEM Sim"
APP_ID = "LTEMSim"


def app_data_dir() -> Path:
    """Per-user state (job history, logs, settings that are not in QSettings)."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    d = base / APP_ID
    (d / "logs").mkdir(parents=True, exist_ok=True)
    (d / "specs").mkdir(parents=True, exist_ok=True)
    return d


def python_exe() -> str:
    """Interpreter for simulation subprocesses: the one running the GUI (the project venv)."""
    return sys.executable


def new_run_dir(kind: str) -> Path:
    """A fresh, not yet created output folder under results/gui_runs/."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = GUI_RUNS / f"{kind}_{stamp}"
    i = 2
    while d.exists():
        d = GUI_RUNS / f"{kind}_{stamp}_{i}"
        i += 1
    return d


def rel(path) -> str:
    """Path relative to the repository root when it lies inside it (for display and for script arguments)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def absolute(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)
