# GitHub Actions Workflow

This project now includes a GitHub Actions workflow at `.github/workflows/top100_pipeline.yml`.

## Supported Modes

- `full`
  Runs the regular post-close pipeline with `python daily_job.py --capture-type post_close`.
- `tail_capture`
  Runs the 14:30 intraday pipeline with an auto-generated `--snapshot-time` in `Asia/Shanghai`.
- `recompute`
  Rebuilds local outputs from existing cached data with `python daily_job.py --no-fetch`.
- `backtest`
  Rebuilds the formal backtest summary service outputs from existing `signals` and `followups` without fetching new market data.
- `tests`
  Runs the unittest suite with `python -m unittest discover -s tests -v`.

## Default Schedule

- Weekdays at 14:30 China time: `tail_capture`
- Weekdays at 17:10 China time: `full`

The workflow uses UTC cron values because GitHub Actions schedules are always defined in UTC:

- `30 6 * * 1-5` -> 14:30 Asia/Shanghai
- `10 9 * * 1-5` -> 17:10 Asia/Shanghai

## What Gets Saved

Each run uploads a workflow artifact that includes:

- `workflow_artifacts/workflow_summary.md`
- workflow logs
- `data/reports/`
- versioned `signals`, `followups`, and `fast_strategy_history` outputs
- `market_regime.csv`
- `popularity_top100.csv`
- `intraday_snapshots.csv`

This keeps the workflow stateless and safe by default. The workflow does not commit generated data back into the repository automatically.

The workflow now also syncs generated outputs back to `main` for these modes:

- `full`
- `tail_capture`
- `recompute`
- `backtest`

It does **not** sync outputs for `tests`.

To avoid endless workflow loops, the `push` trigger ignores report-only commits under:

- `data/raw/popularity_top100.csv`
- `data/raw/intraday_snapshots.csv`
- `data/processed/signals*.csv`
- `data/processed/followups*.csv`
- `data/processed/fast_strategy_history*.csv`
- `data/processed/market_regime.csv`
- `data/reports/*.csv`

## Recommended Usage

If you want a cloud-based replacement for local automations:

1. Push this project to GitHub.
2. Open the Actions tab and enable workflows.
3. Use `workflow_dispatch` to test `tail_capture` and `full` manually.
4. After validation, rely on the default schedule.

## Important Note

Artifacts are still stored on GitHub Actions, but the main daily outputs are now also committed back to `main`. Your local Streamlit dashboard can use them after a normal `git pull`.
