"""Checks that the data-quality checks actually catch what they claim to.

A validation suite that silently passes on broken data is worse than none, so
each check here is shown a deliberately corrupted table and must flag it.
"""

import pandas as pd
import pytest

from football_xg import validate


@pytest.fixture
def shots():
    """A minimal well-formed shots table: two shots, one goal, one fixture."""
    return pd.DataFrame(
        [
            {
                "id": "shot-1",
                "match_id": 1,
                "team": "Alpha",
                "home_team": "Alpha",
                "away_team": "Beta",
                "x": 110.0,
                "y": 40.0,
                "shot_body_part": "Right Foot",
                "has_freeze_frame": True,
                "is_goal": 1,
            },
            {
                "id": "shot-2",
                "match_id": 1,
                "team": "Beta",
                "home_team": "Alpha",
                "away_team": "Beta",
                "x": 100.0,
                "y": 30.0,
                "shot_body_part": "Head",
                "has_freeze_frame": True,
                "is_goal": 0,
            },
        ]
    )


@pytest.fixture
def matches():
    return pd.DataFrame([{"match_id": 1, "home_score": 1, "away_score": 0}])


@pytest.fixture
def events():
    return pd.DataFrame([{"match_id": 1, "type": "Shot"}])


def severities(findings, check):
    return {f.severity for f in findings if f.check == check}


# --------------------------------------------------------------------------
def test_leakage_check_catches_a_post_shot_field(shots):
    shots["shot_end_location"] = [[120.0, 40.0], [120.0, 38.0]]
    findings = validate.check_no_leakage_columns(shots)
    assert findings[0].severity == "error"
    assert "shot_end_location" in findings[0].message


def test_leakage_check_catches_the_raw_target(shots):
    shots["shot_outcome"] = ["Goal", "Saved"]
    assert validate.check_no_leakage_columns(shots)[0].severity == "error"


def test_leakage_check_passes_a_clean_table(shots):
    assert validate.check_no_leakage_columns(shots)[0].severity == "ok"


def test_team_check_catches_a_team_outside_its_fixture(shots):
    shots.loc[0, "team"] = "Gamma"
    finding = validate.check_team_consistency(shots)[0]
    assert finding.severity == "error"
    assert "Gamma" in finding.message


def test_team_check_passes_a_consistent_table(shots):
    assert validate.check_team_consistency(shots)[0].severity == "ok"


def test_location_check_catches_a_shot_off_the_pitch(shots):
    shots.loc[0, "x"] = 130.0
    assert "error" in severities(validate.check_shot_locations(shots), "locations")


def test_orphan_shots_are_an_error(matches, shots):
    shots.loc[0, "match_id"] = 999
    assert "error" in severities(validate.check_match_coverage(matches, shots), "match_ids")


def test_duplicate_ids_are_an_error(shots):
    duped = pd.concat([shots, shots.head(1)], ignore_index=True)
    assert validate.check_duplicate_shots(duped)[0].severity == "error"


def test_missing_required_field_is_an_error(shots):
    shots.loc[0, "x"] = None
    assert "error" in severities(validate.check_missing_values(shots), "missing")


def test_absent_freeze_frames_are_warned_about(shots):
    shots.loc[0, "has_freeze_frame"] = False
    assert "warn" in severities(validate.check_missing_values(shots), "missing")


def test_scores_reconcile_when_shots_explain_the_scoreline(matches, shots, events):
    finding = validate.check_scores_reconcile(matches, shots, events)[0]
    assert finding.severity == "ok"


def test_own_goals_close_the_gap(matches, shots, events):
    """A scoreline goal that is not a shot must be explained by an own goal."""
    matches.loc[0, "away_score"] = 1  # 1-1, but only one shot-goal exists
    assert validate.check_scores_reconcile(matches, shots, events)[0].severity == "error"

    events = pd.concat(
        [events, pd.DataFrame([{"match_id": 1, "type": "Own Goal For"}])],
        ignore_index=True,
    )
    assert validate.check_scores_reconcile(matches, shots, events)[0].severity == "ok"


def test_a_missing_shot_goal_is_caught(matches, shots, events):
    """The check's real job: noticing when shots go missing from the pipeline."""
    assert validate.check_scores_reconcile(matches, shots.iloc[1:], events)[0].severity == "error"
