"""Central configuration: filesystem paths, pitch geometry, and reproducibility settings.

Everything downstream (feature engineering, EDA, models) imports its paths and
constants from here rather than hardcoding them, so the project can be relocated
or rescoped by editing a single file.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
# config.py -> football_xg -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create the data/model/report directories if they do not already exist.

    Safe to call repeatedly; git does not track empty directories, so a fresh
    clone needs this before anything can be written.
    """
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Pitch geometry (StatsBomb coordinate system)
# --------------------------------------------------------------------------
# StatsBomb pitches are normalised to 120x80 regardless of the real stadium
# dimensions. The attacking direction is always towards increasing x, so the
# goal being attacked sits at x = 120.
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

GOAL_CENTER = (120.0, 40.0)
GOAL_POST_LEFT = (120.0, 36.0)
GOAL_POST_RIGHT = (120.0, 44.0)
GOAL_WIDTH = GOAL_POST_RIGHT[1] - GOAL_POST_LEFT[1]  # 8.0

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42
