# Expected Goals (xG) & Match Outcome Prediction
*A High-Level Machine Learning Project on Real Football Event Data*

Project Plan | Prepared by Yoosuf Ahamed | July 2026

---

## 1. Project Overview

This project builds a two-stage machine learning system for football analytics: an Expected Goals (xG) model that scores the quality of individual shots, and a match outcome prediction model that aggregates team-level performance to predict win/draw/loss results. Unlike a typical training exercise, this project is built entirely on real, publicly released match event data and is designed to demonstrate a complete ML lifecycle: data sourcing, feature engineering, classical and deep learning modeling, rigorous evaluation, and (optionally) deployment.

The project is intended as a portfolio-grade deliverable, showcasing skills across tabular ML, sequence modeling with PyTorch, statistical evaluation, and applied sports analytics.

## 2. Objectives

- Source and process a real, open football event dataset (StatsBomb Open Data).
- Engineer shot- and match-level features grounded in football domain knowledge.
- Build a baseline xG model using gradient boosting (LightGBM) with SHAP-based interpretability.
- Build a PyTorch sequence model that scores shots using the buildup sequence of prior events.
- Aggregate shot-level predictions into team match statistics and predict match outcomes.
- Benchmark match outcome predictions against real bookmaker odds where available.
- Produce a clean, reproducible codebase with visualizations suitable for a portfolio write-up.

## 3. Dataset

### 3.1 Source
StatsBomb Open Data (github.com/statsbomb/open-data) — free, event-level football data released publicly for research and analysis. No authentication required for the open subset.

### 3.2 Structure

| File / Folder | Contents |
|---|---|
| `competitions.json` | List of available competitions and seasons |
| `matches/` | Match metadata per competition/season |
| `events/` | Event-level data per match (passes, shots, duels, pressures, etc.) |
| `lineups/` | Team lineups and player details per match |
| `three-sixty/` | 360° freeze-frame data for selected matches |

### 3.3 Access Method
- Python access via the `statsbombpy` package, or direct parsing of the raw JSON files.
- Attribution to StatsBomb is required in any published research or analysis.

### 3.4 Scope for This Project
Initial scope: one or two seasons of a single competition with rich event coverage (e.g., a La Liga or major international tournament season) to keep the dataset manageable while still providing thousands of shots and matches for modeling.

## 4. Methodology

### 4.1 Data Preparation & Exploratory Analysis
- Parse raw JSON event data into structured tables (events, shots, matches).
- Validate schema, check for missing values and inconsistent match/team IDs.
- Exploratory analysis: shot maps, pitch heatmaps, goal conversion rates by zone.

### 4.2 Feature Engineering
- **Shot-level:** distance to goal, shot angle, body part, shot type, defensive pressure, number of defenders in the shot cone.
- **Sequence-level:** the last N events preceding a shot (pass sequences, dribbles, turnovers) as input to a sequence model.
- **Match-level:** aggregated xG for/against, possession share, pressing intensity, historical team form.

### 4.3 Modeling

| Stage | Model | Purpose |
|---|---|---|
| Baseline xG | LightGBM / XGBoost | Predict goal probability from shot features; establish an interpretable benchmark |
| Sequence xG | PyTorch (MLP or attention-based sequence model) | Predict goal probability using the pre-shot event sequence, capturing buildup context |
| Match Outcome | Gradient boosting or PyTorch MLP on aggregated features | Predict win / draw / loss per match |

### 4.4 Evaluation
- Time-respecting train/validation/test splits (train on earlier matches, test on later ones) to avoid leakage.
- Metrics: log loss, Brier score, and calibration curves for probability outputs (accuracy alone is misleading given class imbalance).
- SHAP analysis for the baseline model to explain feature contributions.
- Match outcome predictions benchmarked against real bookmaker odds as an external baseline.

### 4.5 Deployment (Stretch Goal)
- Wrap the trained xG model in a FastAPI endpoint accepting shot coordinates and context, returning a goal probability.
- Optional lightweight front-end or Streamlit demo for interactive shot map exploration.

## 5. Technical Stack

| Layer | Tools |
|---|---|
| Data access | statsbombpy, pandas |
| Classical ML | scikit-learn, LightGBM / XGBoost |
| Deep learning | PyTorch |
| Interpretability | SHAP |
| Visualization | matplotlib, mplsoccer |
| Serving (optional) | FastAPI |
| Experiment tracking (optional) | MLflow or Weights & Biases |

## 6. Milestones & Timeline

| Milestone | Description |
|---|---|
| 1. Data Ingestion & EDA | Load StatsBomb data, build shot/match tables, produce shot maps and pitch heatmaps |
| 2. Feature Engineering | Build shot-level and sequence-level feature sets |
| 3. Baseline xG Model | Train and calibrate LightGBM model, run SHAP analysis |
| 4. PyTorch Sequence Model | Build and train a sequence model over pre-shot events, compare against baseline |
| 5. Match Outcome Model | Aggregate to match level, train outcome model, benchmark vs. bookmaker odds |
| 6. Write-up & Demo | Portfolio write-up with visualizations; optional FastAPI demo endpoint |

## 7. Deliverables

- A reproducible codebase (data pipeline, feature engineering, model training, evaluation).
- A baseline xG model and a PyTorch sequence-based xG model, compared side by side.
- A match outcome model benchmarked against bookmaker odds.
- Visualizations: shot maps, calibration curves, SHAP summary plots.
- A written project summary suitable for a portfolio or CV.
- (Stretch) A working FastAPI demo endpoint.

## 8. Risks & Considerations

- **Data leakage:** care must be taken not to use post-shot information when predicting shot outcomes.
- **Class imbalance:** most shots do not result in goals, requiring careful metric selection and calibration.
- **Dataset scope:** StatsBomb's free tier covers a limited set of competitions/seasons; scope should be chosen to ensure sufficient sample size.
- **Attribution requirement:** StatsBomb must be credited in any published output.
