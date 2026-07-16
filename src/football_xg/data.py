"""Ingestion of StatsBomb Open Data, with an on-disk cache of the raw JSON.

statsbombpy fetches from GitHub on every call and only caches in a temp directory
for ten minutes, so a full season is ~380 HTTP requests that vanish between runs.
This module puts a durable cache in front of it: every response is written to
data/raw/ exactly as StatsBomb published it, and re-read from there afterwards.

Caching the raw JSON rather than a parsed DataFrame is deliberate:

* data/raw/ then holds genuinely raw data, and everything else is reproducible
  from a cold start.
* Parquet silently coerces the nested `shot_freeze_frame` lists into numpy
  arrays, so a parsed cache would not round-trip the one field the defensive
  features depend on.

The cache is installed by swapping statsbombpy's single HTTP entry point rather
than by reimplementing its parsing. Turning raw StatsBomb JSON into a flat frame
is fiddly -- type-specific attributes get hoisted into `shot_*` / `pass_*`
columns, and skipping that step yields a table that looks plausible but is
missing most of its columns. Intercepting the fetch means we inherit their
parsing exactly and only own the caching.

Data provided by StatsBomb (https://github.com/statsbomb/open-data).
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests
from statsbombpy import public, sb
from statsbombpy.config import OPEN_DATA_PATHS

from football_xg import config

logger = logging.getLogger(__name__)

# StatsBomb's raw JSON is served straight off GitHub, which rate-limits abusive
# clients. Keep the pool small and identify ourselves.
_MAX_WORKERS = 6
_HEADERS = {"User-Agent": "football-xg/0.1 (portfolio project; StatsBomb Open Data)"}
_TIMEOUT = 30

# Every open-data URL is this prefix plus a path that mirrors the repo tree.
_OPEN_DATA_PREFIX = "/open-data/master/data/"

# Taken from statsbombpy rather than hardcoded, so the prefetch below cannot
# drift from the URL the library itself would request.
_EVENTS_URL = OPEN_DATA_PATHS["events"]


# --------------------------------------------------------------------------
# Raw JSON cache
# --------------------------------------------------------------------------
def _cache_path(url: str) -> Path:
    """Map a StatsBomb open-data URL to its file in data/raw/.

    The cache mirrors the open-data repo's own layout (`events/<match_id>.json`,
    `matches/<competition_id>/<season_id>.json`), so anyone who knows that repo
    can read our cache without a key.
    """
    try:
        relative = url.split(_OPEN_DATA_PREFIX, 1)[1]
    except IndexError:
        raise ValueError(f"not a StatsBomb open-data URL: {url!r}") from None
    return config.RAW_DATA_DIR / relative


def _cached_get_response(url: str) -> list | dict:
    """statsbombpy's fetch, backed by data/raw/.

    Writes are atomic (temp file, then rename) so an interrupted download cannot
    leave a truncated file that later runs would happily read as valid JSON.
    """
    path = _cache_path(url)
    if path.is_file():
        return json.loads(path.read_text())

    response = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    response.raise_for_status()
    payload = response.json()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)
    return payload


@contextmanager
def _cached_open_data() -> Iterator[None]:
    """Point statsbombpy's HTTP fetch at our cache for the duration of a call.

    `public.get_response` is the one place the library reaches the network for
    open data, which makes it a narrow seam. Patched per-call rather than at
    import so simply importing this module has no global side effect.
    """
    original = public.get_response
    public.get_response = _cached_get_response
    try:
        yield
    finally:
        public.get_response = original


def _uncached(urls: list[str]) -> list[str]:
    return [u for u in urls if not _cache_path(u).is_file()]


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_competitions() -> pd.DataFrame:
    """Every competition-season in the open-data release."""
    with _cached_open_data():
        return sb.competitions()


def load_matches(
    competition_id: int = config.COMPETITION_ID,
    season_id: int = config.SEASON_ID,
) -> pd.DataFrame:
    """Match metadata for one competition-season, sorted chronologically.

    Sorting here rather than at the call site means downstream code can rely on
    row order reflecting real-world order, which is what the time-respecting
    splits are built on.
    """
    with _cached_open_data():
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    return matches.sort_values("match_date").reset_index(drop=True)


def load_events(match_id: int) -> pd.DataFrame:
    """Every event in a single match, flattened one column per attribute."""
    with _cached_open_data():
        return sb.events(match_id=match_id)


def load_season_events(
    competition_id: int = config.COMPETITION_ID,
    season_id: int = config.SEASON_ID,
) -> pd.DataFrame:
    """Every event in every match of a competition-season, as one table.

    The first call downloads a few hundred JSON files and takes minutes; later
    calls read the cache and take seconds. The download is threaded because the
    bottleneck is network latency, not CPU, but parsing stays sequential.
    """
    matches = load_matches(competition_id, season_id)
    match_ids = matches["match_id"].astype(int).tolist()

    urls = [_EVENTS_URL.format(match_id=m) for m in match_ids]
    missing = _uncached(urls)
    if missing:
        logger.info("downloading events for %d matches", len(missing))
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            list(pool.map(_cached_get_response, missing))

    return pd.concat([load_events(m) for m in match_ids], ignore_index=True)


# --------------------------------------------------------------------------
# Shot table
# --------------------------------------------------------------------------
# Fields StatsBomb records on a shot that describe what happened *after* the
# ball was struck. An xG model predicts the outcome of a shot from what the
# player could see when they hit it, so every one of these is leakage and none
# may become a feature. They are dropped at ingestion rather than filtered at
# training time, so a later feature can never reach them by accident.
POST_SHOT_FIELDS = (
    "shot_end_location",  # where the ball finished -- in the net, or not
    "shot_deflected",  # a deflection happens after contact
    "shot_saved_off_target",  # implies a save, so implies no goal
    "shot_saved_to_post",
)

# StatsBomb's own xG. Kept as a benchmark to measure our model against, never as
# a feature -- training on it would just teach our model to copy theirs.
BENCHMARK_FIELD = "shot_statsbomb_xg"

# Attributes true at the moment of contact, and so fair game as features.
# StatsBomb writes these as `True` or omits them, so absence means False.
_PRE_SHOT_FLAGS = (
    "under_pressure",
    "shot_first_time",
    "shot_one_on_one",
    "shot_open_goal",
    "shot_aerial_won",
    "shot_follows_dribble",
    "shot_redirect",
)

_SHOT_COLUMNS = (
    # identity and ordering
    "id",
    "match_id",
    "index",
    "period",
    "minute",
    "second",
    "timestamp",
    "possession",
    "possession_team",
    "play_pattern",
    # who
    "team",
    "player",
    "position",
    # where -- raw coordinates; derived geometry is the feature layer's job
    "x",
    "y",
    # how
    "shot_type",
    "shot_body_part",
    "shot_technique",
    "shot_freeze_frame",
    "shot_key_pass_id",
)


def build_shots_table(
    events: pd.DataFrame | None = None,
    matches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per shot, carrying only information available before contact.

    Post-shot fields are dropped here (see POST_SHOT_FIELDS) and the target is
    reduced to a single boolean. Penalties are kept but flagged: they are a
    degenerate cluster -- same spot, no defenders, ~71% conversion -- so the
    modelling layer excludes them, but the EDA should still be able to see them.

    Returns a table sorted by match date then within-match event order, so a
    time-respecting split is a row slice.
    """
    if events is None:
        events = load_season_events()
    if matches is None:
        matches = load_matches()

    shots = events[events["type"] == "Shot"].copy()

    # `location` is a [x, y] list; split it so downstream code can do arithmetic.
    coords = pd.DataFrame(
        shots["location"].tolist(), index=shots.index, columns=["x", "y"]
    )
    shots[["x", "y"]] = coords

    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
    shots["is_penalty"] = shots["shot_type"] == "Penalty"
    # 81 of ~9k shots have no freeze frame, so the defensive features cannot be
    # assumed present. Flagging it here keeps that decision explicit downstream.
    shots["has_freeze_frame"] = shots["shot_freeze_frame"].notna()

    for flag in _PRE_SHOT_FLAGS:
        shots[flag] = shots[flag].fillna(False).astype(bool) if flag in shots else False

    keep = [
        *_SHOT_COLUMNS,
        *_PRE_SHOT_FLAGS,
        "is_penalty",
        "has_freeze_frame",
        BENCHMARK_FIELD,
        "is_goal",
    ]
    shots = shots[[c for c in keep if c in shots.columns]]

    # Attach match context: the date drives the time-respecting split, and
    # home/away decides which side of a fixture a shot belongs to.
    context = matches[["match_id", "match_date", "home_team", "away_team"]]
    shots = shots.merge(context, on="match_id", how="left")
    shots["is_home"] = shots["team"] == shots["home_team"]

    return shots.sort_values(["match_date", "match_id", "index"]).reset_index(drop=True)
