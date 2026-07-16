"""Data quality checks on the ingested StatsBomb tables.

Open data is real data, which means it has real defects: missing freeze frames,
shots recorded outside the pitch, scorelines that disagree with the events. This
module states what we expect to be true and reports where it is not, rather than
asserting and dying -- most of these findings are things to handle in the feature
layer, not reasons to stop.

Each check returns a list of Finding. `run_all_checks` runs the suite and prints
a report; `notebooks/01_data_survey.ipynb` renders the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from football_xg import config, data

Severity = Literal["ok", "note", "warn", "error"]


@dataclass(frozen=True)
class Finding:
    """One quality observation about a table."""

    check: str
    severity: Severity
    message: str

    def __str__(self) -> str:
        marker = {"ok": "OK  ", "note": "NOTE", "warn": "WARN", "error": "FAIL"}
        return f"[{marker[self.severity]}] {self.check}: {self.message}"


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def check_no_leakage_columns(shots: pd.DataFrame) -> list[Finding]:
    """The shots table must not carry any post-shot field.

    This is the check that matters most in the whole module. If a post-shot
    field survives ingestion, every metric downstream is quietly inflated and
    the model looks brilliant for the wrong reason.
    """
    leaked = [c for c in data.POST_SHOT_FIELDS if c in shots.columns]
    if leaked:
        return [
            Finding(
                "leakage",
                "error",
                f"post-shot fields present in shots table: {leaked}",
            )
        ]
    if "shot_outcome" in shots.columns:
        return [
            Finding("leakage", "error", "shot_outcome survived; it is the target")
        ]
    return [Finding("leakage", "ok", "no post-shot fields in the shots table")]


def check_match_coverage(matches: pd.DataFrame, shots: pd.DataFrame) -> list[Finding]:
    """Every match should appear, and every shot should belong to a real match."""
    findings = []
    match_ids = set(matches["match_id"])
    shot_match_ids = set(shots["match_id"])

    orphans = shot_match_ids - match_ids
    if orphans:
        findings.append(
            Finding(
                "match_ids",
                "error",
                f"{len(orphans)} shots reference match_ids absent from the match table",
            )
        )
    else:
        findings.append(
            Finding("match_ids", "ok", "every shot maps to a known match")
        )

    silent = match_ids - shot_match_ids
    if silent:
        findings.append(
            Finding("match_ids", "warn", f"{len(silent)} matches contain no shots")
        )
    return findings


def check_team_consistency(shots: pd.DataFrame) -> list[Finding]:
    """A shot's team must be one of the two teams in that fixture.

    Reads home_team/away_team off the shots table, which is where the match
    merge deposited them -- so this catches a bad merge as well as a mislabelled
    event. If it fails, home/away and every match-level aggregate built on it
    is wrong.
    """
    mismatched = shots[
        (shots["team"] != shots["home_team"]) & (shots["team"] != shots["away_team"])
    ]
    if len(mismatched):
        teams = mismatched["team"].unique()[:5]
        return [
            Finding(
                "team_ids",
                "error",
                f"{len(mismatched)} shots by a team not in their own fixture, e.g. {list(teams)}",
            )
        ]
    return [
        Finding("team_ids", "ok", "every shot's team plays in its own fixture")
    ]


def check_shot_locations(shots: pd.DataFrame) -> list[Finding]:
    """Shot coordinates must sit inside the normalised 120x80 pitch."""
    findings = []
    off_pitch = shots[
        (shots["x"] < 0)
        | (shots["x"] > config.PITCH_LENGTH)
        | (shots["y"] < 0)
        | (shots["y"] > config.PITCH_WIDTH)
    ]
    if len(off_pitch):
        findings.append(
            Finding(
                "locations",
                "error",
                f"{len(off_pitch)} shots outside the {config.PITCH_LENGTH:.0f}x{config.PITCH_WIDTH:.0f} pitch",
            )
        )
    else:
        findings.append(Finding("locations", "ok", "all shots inside the pitch"))

    # A shot from the defensive half is legal but rare; a cluster of them means
    # the attacking-direction normalisation is wrong.
    own_half = (shots["x"] < config.PITCH_LENGTH / 2).sum()
    share = own_half / len(shots)
    severity: Severity = "warn" if share > 0.01 else "note"
    findings.append(
        Finding(
            "locations",
            severity,
            f"{own_half} shots ({share:.2%}) from the defensive half",
        )
    )
    return findings


def check_missing_values(shots: pd.DataFrame) -> list[Finding]:
    """Report columns with gaps, and confirm the ones we cannot tolerate."""
    findings = []
    required = ["id", "match_id", "team", "x", "y", "shot_body_part", "is_goal"]
    for col in required:
        n = shots[col].isna().sum()
        if n:
            findings.append(
                Finding("missing", "error", f"{col} is null on {n} shots but is required")
            )
    if not findings:
        findings.append(Finding("missing", "ok", "no gaps in required shot fields"))

    no_ff = (~shots["has_freeze_frame"]).sum()
    if no_ff:
        findings.append(
            Finding(
                "missing",
                "warn",
                f"{no_ff} shots ({no_ff / len(shots):.2%}) have no freeze frame; "
                "defensive features cannot be computed for these",
            )
        )
    return findings


def check_class_balance(shots: pd.DataFrame) -> list[Finding]:
    """Goal rate should land near the ~10% that makes accuracy a useless metric."""
    rate = shots["is_goal"].mean()
    severity: Severity = "ok" if 0.05 < rate < 0.20 else "warn"
    return [
        Finding(
            "class_balance",
            severity,
            f"{shots['is_goal'].sum()} goals in {len(shots)} shots ({rate:.2%}); "
            f"a model predicting 'no goal' always would score {1 - rate:.2%} accuracy",
        )
    ]


def check_duplicate_shots(shots: pd.DataFrame) -> list[Finding]:
    """StatsBomb event ids are UUIDs and must be unique across the season."""
    dupes = shots["id"].duplicated().sum()
    if dupes:
        return [Finding("duplicates", "error", f"{dupes} duplicated shot ids")]
    return [Finding("duplicates", "ok", f"all {len(shots)} shot ids unique")]


def check_scores_reconcile(
    matches: pd.DataFrame, shots: pd.DataFrame, events: pd.DataFrame
) -> list[Finding]:
    """Every goal in the scorelines must be explained by the event data.

    The strongest end-to-end check available: it ties our derived tables back to
    an independent field (the recorded result) that nothing in our pipeline
    touches. Shot-goals alone will not reconcile, because an own goal is its own
    event type rather than a shot -- StatsBomb logs each as a pair ("Own Goal
    For" for the beneficiary, "Own Goal Against" for the culprit), so we count
    one side. Anything left over after that means shots are being lost or
    double-counted.
    """
    shot_goals = shots.groupby("match_id")["is_goal"].sum()
    own_goals = (
        events[events["type"] == "Own Goal For"].groupby("match_id").size()
    )

    tally = matches[["match_id", "home_score", "away_score"]].copy()
    tally["recorded"] = tally["home_score"] + tally["away_score"]
    tally["explained"] = (
        tally["match_id"].map(shot_goals).fillna(0)
        + tally["match_id"].map(own_goals).fillna(0)
    )
    residual = (tally["recorded"] - tally["explained"]).abs().sum()

    if residual:
        unexplained = (tally["recorded"] != tally["explained"]).sum()
        return [
            Finding(
                "scores",
                "error",
                f"{int(residual)} goals across {unexplained} matches are not explained "
                "by shots + own goals; events are being lost or double-counted",
            )
        ]
    return [
        Finding(
            "scores",
            "ok",
            f"all {int(tally['recorded'].sum())} goals in {len(tally)} matches reconcile "
            f"with shots + {int(own_goals.sum())} own goals, exactly",
        )
    ]


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run_all_checks(
    matches: pd.DataFrame,
    shots: pd.DataFrame,
    events: pd.DataFrame,
    verbose: bool = True,
) -> list[Finding]:
    """Run every check and return the findings, worst first."""
    findings = [
        *check_no_leakage_columns(shots),
        *check_duplicate_shots(shots),
        *check_match_coverage(matches, shots),
        *check_team_consistency(shots),
        *check_shot_locations(shots),
        *check_missing_values(shots),
        *check_class_balance(shots),
        *check_scores_reconcile(matches, shots, events),
    ]
    order = {"error": 0, "warn": 1, "note": 2, "ok": 3}
    findings.sort(key=lambda f: order[f.severity])

    if verbose:
        for finding in findings:
            print(finding)
        errors = sum(f.severity == "error" for f in findings)
        print(
            f"\n{len(findings)} checks, {errors} failing"
            if errors
            else f"\n{len(findings)} checks, none failing"
        )
    return findings
