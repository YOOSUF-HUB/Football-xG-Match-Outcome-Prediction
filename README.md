# Expected Goals (xG) & Match Outcome Prediction

A two-stage machine learning system for football analytics, built end-to-end on **real** event data:

1. **Expected Goals (xG)** — score the quality of individual shots, using both a gradient-boosted baseline and a PyTorch sequence model that reads the buildup play preceding each shot.
2. **Match outcome prediction** — aggregate shot- and team-level performance to predict win / draw / loss.

No synthetic data. Every number traces back to StatsBomb's publicly released event data.

## Status

Work in progress, built in stages:

- [x] **0. Project scaffolding & environment**
- [x] **1. Data ingestion & EDA** — [`notebooks/01_data_survey.ipynb`](notebooks/01_data_survey.ipynb)
- [ ] **2. Feature engineering** — shot-level, sequence-level, and match-level features
- [ ] **3. Baseline xG model** — LightGBM + calibration + SHAP
- [ ] **4. PyTorch sequence xG model** — attention over pre-shot events, compared to the baseline
- [ ] **5. Match outcome model** — aggregate to match level, benchmark vs. bookmaker odds
- [ ] **6. Write-up & demo** — portfolio write-up; optional FastAPI endpoint

## The data

**La Liga 2015/16** — the complete 380-match season: 1.3M events, 9,168 shots, 1,014 goals.

The choice matters more than it looks. Most La Liga seasons in StatsBomb's free tier are
*Barcelona-only* — every match features one club, because the data was released to showcase
Messi's career. Training an xG model on that teaches it one team's shot profile and calls it
football. The 2015/16 release is one of the few complete league seasons available free: all 20
teams, no team in more than ~10% of matches, and a freeze frame of every player's position on
99.1% of shots. The coverage comparison that drove the decision is the first section of the
survey notebook.

## Setup

Requires Python 3.12.

**macOS only** — LightGBM's wheel links against OpenMP, which macOS does not ship. Install it first, or `import lightgbm` fails with `Library not loaded: @rpath/libomp.dylib`:

```bash
brew install libomp
```

Then create the environment:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

To reproduce the exact versions this project was developed against, use the lock file
instead of `requirements.txt`:

```bash
.venv/bin/pip install -r requirements.lock.txt
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
| `src/football_xg/config.py` | Paths, pitch geometry, competition scope — everything else imports from here |
| `src/football_xg/data.py` | StatsBomb ingestion, on-disk cache, shots table |
| `src/football_xg/validate.py` | Data quality checks, including the leakage guard |
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

- **Leakage.** Shot outcome must be predicted using only information available *before* the ball is struck. Post-shot fields — where the ball ended up, whether it was deflected, whether it was saved — are dropped at ingestion rather than filtered at training time, so no feature can reach them by accident. They're enumerated in `data.POST_SHOT_FIELDS` and a test asserts none survive. Splits are time-respecting: train on earlier matches, test on later ones, never a random shuffle.
- **Class imbalance.** 11.1% of shots are goals, so accuracy is meaningless — predicting "no goal" every time scores 88.9%. Models are judged on log loss, Brier score, and calibration curves. An xG model is only useful if its probabilities are honest.

Two reference points bracket the problem, both measured in the survey notebook:

| | Log loss | Brier |
| --- | --- | --- |
| Predicting the base rate (the floor) | 0.3478 | 0.0984 |
| StatsBomb's own xG (the bar) | 0.2593 | 0.0751 |

StatsBomb ship their xG on every shot. It is **not** a feature — training on it would just teach
our model to imitate theirs. It's well calibrated on this season and its season total lands within
3.4% of actual goals scored, so it's a serious benchmark rather than a strawman.

## Data & attribution

Data provided by **StatsBomb** via their [Open Data](https://github.com/statsbomb/open-data) release, used here under their terms for research and analysis.

## License

Code in this repository is available for review as a portfolio piece. StatsBomb data is subject to the [StatsBomb Open Data terms](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
