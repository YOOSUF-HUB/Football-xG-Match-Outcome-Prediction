"""Checks on ingestion: the cache layout and the shots table's contract.

These build tiny frames by hand rather than hitting the network, so they run in
milliseconds and fail for one reason only. The end-to-end assertion that the
real season parses correctly lives in the validation suite, which runs against
the actual data.
"""

import pandas as pd
import pytest

from football_xg import config, data


# --------------------------------------------------------------------------
# Cache paths
# --------------------------------------------------------------------------
def test_cache_mirrors_the_open_data_repo_layout():
    """A URL maps to the same relative path the open-data repo uses."""
    events = data._cache_path(
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3825562.json"
    )
    assert events == config.RAW_DATA_DIR / "events" / "3825562.json"

    matches = data._cache_path(
        "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/11/27.json"
    )
    assert matches == config.RAW_DATA_DIR / "matches" / "11" / "27.json"


def test_cache_path_rejects_a_foreign_url():
    """Anything not from the open-data tree has no place in data/raw/."""
    with pytest.raises(ValueError):
        data._cache_path("https://example.com/events/1.json")


def test_cached_open_data_restores_the_original_fetch():
    """The patch must not leak out of its context, even if the body raises."""
    from statsbombpy import public

    original = public.get_response
    with data._cached_open_data():
        assert public.get_response is data._cached_get_response
    assert public.get_response is original

    with pytest.raises(RuntimeError):
        with data._cached_open_data():
            raise RuntimeError("boom")
    assert public.get_response is original


# --------------------------------------------------------------------------
# Shots table
# --------------------------------------------------------------------------
@pytest.fixture
def fake_events():
    """Two shots and a pass, shaped like statsbombpy's flattened output."""
    return pd.DataFrame(
        [
            {
                "id": "shot-1",
                "match_id": 1,
                "index": 10,
                "type": "Shot",
                "team": "Alpha",
                "player": "A. Striker",
                "location": [110.0, 40.0],
                "shot_outcome": "Goal",
                "shot_type": "Open Play",
                "shot_body_part": "Right Foot",
                "shot_statsbomb_xg": 0.4,
                "shot_end_location": [120.0, 40.0, 1.0],
                "shot_deflected": None,
                "shot_freeze_frame": [{"location": [115.0, 40.0], "teammate": False}],
                "under_pressure": True,
                "shot_first_time": None,
            },
            {
                "id": "shot-2",
                "match_id": 1,
                "index": 20,
                "type": "Shot",
                "team": "Beta",
                "player": "B. Forward",
                "location": [100.0, 30.0],
                "shot_outcome": "Saved",
                "shot_type": "Penalty",
                "shot_body_part": "Left Foot",
                "shot_statsbomb_xg": 0.76,
                "shot_end_location": [120.0, 38.0, 0.5],
                "shot_deflected": None,
                "shot_freeze_frame": None,
                "under_pressure": None,
                "shot_first_time": True,
            },
            {
                "id": "pass-1",
                "match_id": 1,
                "index": 5,
                "type": "Pass",
                "team": "Alpha",
                "location": [60.0, 40.0],
            },
        ]
    )


@pytest.fixture
def fake_matches():
    return pd.DataFrame(
        [
            {
                "match_id": 1,
                "match_date": pd.Timestamp("2015-08-21"),
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_score": 1,
                "away_score": 0,
            }
        ]
    )


def test_shots_table_keeps_only_shots(fake_events, fake_matches):
    shots = data.build_shots_table(fake_events, fake_matches)
    assert len(shots) == 2
    assert set(shots["id"]) == {"shot-1", "shot-2"}


def test_shots_table_drops_every_post_shot_field(fake_events, fake_matches):
    """The leakage guarantee: post-shot fields must not survive ingestion."""
    shots = data.build_shots_table(fake_events, fake_matches)
    for field in data.POST_SHOT_FIELDS:
        assert field not in shots.columns, f"{field} leaked into the shots table"
    assert "shot_outcome" not in shots.columns, "the raw target leaked"


def test_shots_table_keeps_the_benchmark_but_not_as_a_hidden_target(
    fake_events, fake_matches
):
    """StatsBomb's xG is retained deliberately -- to compare against, not train on."""
    shots = data.build_shots_table(fake_events, fake_matches)
    assert data.BENCHMARK_FIELD in shots.columns


def test_location_is_split_into_coordinates(fake_events, fake_matches):
    shots = data.build_shots_table(fake_events, fake_matches).set_index("id")
    assert shots.loc["shot-1", "x"] == 110.0
    assert shots.loc["shot-1", "y"] == 40.0


def test_target_is_derived_from_outcome(fake_events, fake_matches):
    shots = data.build_shots_table(fake_events, fake_matches).set_index("id")
    assert shots.loc["shot-1", "is_goal"] == 1
    assert shots.loc["shot-2", "is_goal"] == 0


def test_absent_flags_become_false_not_null(fake_events, fake_matches):
    """StatsBomb omits a flag rather than writing False; the table must not."""
    shots = data.build_shots_table(fake_events, fake_matches).set_index("id")
    assert shots.loc["shot-1", "under_pressure"] is True or shots.loc["shot-1", "under_pressure"]
    assert not shots.loc["shot-2", "under_pressure"]
    assert not shots.loc["shot-1", "shot_first_time"]
    assert shots["under_pressure"].notna().all()


def test_penalties_are_flagged_not_dropped(fake_events, fake_matches):
    """Ingestion preserves; the modelling layer is what excludes penalties."""
    shots = data.build_shots_table(fake_events, fake_matches).set_index("id")
    assert not shots.loc["shot-1", "is_penalty"]
    assert shots.loc["shot-2", "is_penalty"]


def test_missing_freeze_frame_is_flagged(fake_events, fake_matches):
    shots = data.build_shots_table(fake_events, fake_matches).set_index("id")
    assert shots.loc["shot-1", "has_freeze_frame"]
    assert not shots.loc["shot-2", "has_freeze_frame"]


def test_home_away_is_resolved_from_the_fixture(fake_events, fake_matches):
    shots = data.build_shots_table(fake_events, fake_matches).set_index("id")
    assert shots.loc["shot-1", "is_home"]  # Alpha are at home
    assert not shots.loc["shot-2", "is_home"]  # Beta are away


def test_shots_are_ordered_for_time_respecting_splits(fake_events, fake_matches):
    """Row order must reflect real-world order so a split can be a row slice."""
    shots = data.build_shots_table(fake_events, fake_matches)
    assert shots["match_date"].is_monotonic_increasing
    assert list(shots["index"]) == sorted(shots["index"])
