# Expected Goals (xG) & Match Outcome Prediction

A two-stage machine learning system for football analytics, built end-to-end on **real** event data:

1. **Expected Goals (xG)** — score the quality of individual shots, using both a gradient-boosted baseline and a PyTorch sequence model that reads the buildup play preceding each shot.
2. **Match outcome prediction** — aggregate shot- and team-level performance to predict win / draw / loss.

No synthetic data. Every number traces back to StatsBomb's publicly released event data.

## Status

Work in progress, built in stages:

- [x] **0. Project scaffolding & environment**
- [ ] **1. Data ingestion & EDA** — load StatsBomb data, build shot/match tables, shot maps and pitch heatmaps
- [ ] **2. Feature engineering** — shot-level, sequence-level, and match-level features
- [ ] **3. Baseline xG model** — LightGBM + calibration + SHAP
- [ ] **4. PyTorch sequence xG model** — attention over pre-shot events, compared to the baseline
- [ ] **5. Match outcome model** — aggregate to match level, benchmark vs. bookmaker odds
- [ ] **6. Write-up & demo** — portfolio write-up; optional FastAPI endpoint

## Setup

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

Register the environment as a Jupyter kernel so the notebooks use it:

```bash
.venv/bin/python -m ipykernel install --user \
  --name football-xg --display-name "Python (football-xg)"
```

Verify the install:

```bash
.venv/bin/pytest
```

## Project layout

| Path | Contents |
| --- | --- |
| `src/football_xg/` | All reusable code: data loading, features, models, training, evaluation |
| `notebooks/` | Exploratory analysis only — findings get promoted into `src/` |
| `data/raw/` | Cached StatsBomb JSON (git-ignored, reproducible) |
| `data/processed/` | Derived parquet tables (git-ignored, reproducible) |
| `models/` | Trained model artifacts (git-ignored) |
| `reports/figures/` | Generated plots (git-ignored) |
| `tests/` | Test suite |

Data and artifacts are deliberately kept out of git — everything under `data/` and
`models/` can be regenerated from the pipeline in `src/`.

## Methodology notes

Two concerns drive most of the design decisions in this repo:

- **Leakage.** Shot outcome must be predicted using only information available *before* the ball is struck. Post-shot fields (shot outcome, end location, goalkeeper reaction) are excluded from features, and splits are time-respecting — train on earlier matches, test on later ones, never a random shuffle.
- **Class imbalance.** Roughly 1 shot in 10 is a goal, so accuracy is close to meaningless. Models are judged on log loss, Brier score, and calibration curves — an xG model is only useful if its probabilities are honest.

## Data & attribution

Data provided by **StatsBomb** via their [Open Data](https://github.com/statsbomb/open-data) release, used here under their terms for research and analysis.

## License

Code in this repository is available for review as a portfolio piece. StatsBomb data is subject to the [StatsBomb Open Data terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
