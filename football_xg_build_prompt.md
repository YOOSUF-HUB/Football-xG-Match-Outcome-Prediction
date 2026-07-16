# Build Prompt: Football xG & Match Outcome Prediction (PyTorch)

Use this prompt with an AI coding assistant (e.g. Claude Code) to scaffold and build the project end-to-end.

---

## Prompt

You are helping me build a portfolio-grade machine learning project: an **Expected Goals (xG) model** and a **match outcome prediction model** for football, built on **real** event data from StatsBomb's open dataset (no synthetic data). The stack must include **PyTorch**. Treat this as a real project, not a toy notebook — proper structure, reproducibility, and evaluation rigor matter more than raw model accuracy.

### Context
- Data source: StatsBomb Open Data (`https://github.com/statsbomb/open-data`), accessed via the `statsbombpy` Python package or by parsing the raw JSON directly. No auth required for the open subset. Attribution to StatsBomb is required in any output.
- Scope: start with one or two seasons of a single competition with strong event coverage (e.g. a La Liga season or a major international tournament), to keep volume manageable while still yielding thousands of shots and hundreds of matches.

### What to build

1. **Project structure** — a clean, reproducible repo layout: `data/`, `notebooks/` (for EDA only), `src/` (data loading, feature engineering, models, training, evaluation), `models/` (saved artifacts), `reports/` (figures, write-up), `tests/`, `requirements.txt` or `pyproject.toml`, and a top-level `README.md`.

2. **Data ingestion** — pull competitions/matches/events from StatsBomb Open Data, parse into structured pandas tables for events, shots, and matches. Validate schema, check for missing values and inconsistent match/team IDs.

3. **Exploratory analysis** — shot maps and pitch heatmaps (use `mplsoccer`), goal conversion rates by pitch zone, class balance check on shot outcomes.

4. **Feature engineering**
   - Shot-level: distance to goal, shot angle, body part, shot type, defensive pressure, defenders in the shot cone.
   - Sequence-level: the last N events preceding each shot (passes, dribbles, turnovers), structured for a sequence model.
   - Match-level: aggregated xG for/against, possession share, pressing intensity, historical team form.

5. **Modeling**
   - Baseline xG model: LightGBM or XGBoost on shot-level features, with SHAP for interpretability.
   - Sequence xG model: a PyTorch model (start with an MLP, then an attention-based sequence model) that consumes the pre-shot event sequence and predicts goal probability. Compare against the baseline.
   - Match outcome model: aggregate shot/team stats to match level and predict win/draw/loss, using gradient boosting or a PyTorch MLP.

6. **Evaluation**
   - Time-respecting splits (train on earlier matches, test on later ones) — no random shuffling, to avoid leakage.
   - Metrics: log loss, Brier score, calibration curves (not just accuracy, given class imbalance).
   - Benchmark the match outcome model against real bookmaker odds if available.

7. **Optional stretch goal** — wrap the trained xG model in a FastAPI endpoint that accepts shot coordinates/context and returns a goal probability.

### How to work with me
- Build incrementally: data ingestion and EDA first, then features, then the baseline model, then the PyTorch sequence model, then the match outcome model, then evaluation and write-up. Confirm each stage works (e.g. show sample output, a plot, or a metric) before moving to the next.
- Flag any data leakage risks explicitly as you design features and splits.
- Keep the code modular and documented so it reads as a real engineering project, not a script.
- At the end, help me draft a short project write-up (problem, approach, results, what I'd do with more time) suitable for a portfolio/CV.

---

*Reference: full project plan available in `football_xg_project_plan.md` / `.docx`.*
