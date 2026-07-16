"""Checks on the project's paths and pitch constants.

These are cheap guards against the two ways config.py silently breaks: the
PROJECT_ROOT parent-walk drifting if the module is moved, and the pitch
geometry being edited into an inconsistent state.
"""

from football_xg import config


def test_project_root_is_the_repo_root():
    """PROJECT_ROOT walks up from src/football_xg/config.py to the repo root."""
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()
    assert (config.PROJECT_ROOT / "src" / "football_xg").is_dir()


def test_data_dirs_sit_under_the_project_root():
    assert config.RAW_DATA_DIR.parent == config.DATA_DIR
    assert config.PROCESSED_DATA_DIR.parent == config.DATA_DIR
    assert config.DATA_DIR.parent == config.PROJECT_ROOT
    assert config.FIGURES_DIR.parent == config.REPORTS_DIR


def test_ensure_dirs_is_idempotent(tmp_path, monkeypatch):
    """ensure_dirs() must succeed when the directories already exist."""
    for attr in ("RAW_DATA_DIR", "PROCESSED_DATA_DIR", "MODELS_DIR", "REPORTS_DIR", "FIGURES_DIR"):
        monkeypatch.setattr(config, attr, tmp_path / attr.lower())

    config.ensure_dirs()
    config.ensure_dirs()  # second call must not raise

    assert (tmp_path / "raw_data_dir").is_dir()
    assert (tmp_path / "figures_dir").is_dir()


def test_goal_sits_on_the_attacking_byline():
    """StatsBomb normalises to 120x80 with the attacked goal at x=120."""
    assert config.GOAL_CENTER == (config.PITCH_LENGTH, config.PITCH_WIDTH / 2)
    assert config.GOAL_POST_LEFT[0] == config.PITCH_LENGTH
    assert config.GOAL_POST_RIGHT[0] == config.PITCH_LENGTH


def test_goal_is_eight_units_wide_and_centred():
    """A real goal is 8 yards wide; StatsBomb's y-axis is 1 unit per yard here."""
    assert config.GOAL_WIDTH == 8.0
    midpoint = (config.GOAL_POST_LEFT[1] + config.GOAL_POST_RIGHT[1]) / 2
    assert midpoint == config.GOAL_CENTER[1]
