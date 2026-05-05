# GitHub Actions Workflow

This project now includes a GitHub Actions workflow at `.github/workflows/top100_pipeline.yml`.

## Supported Modes

- `full`
  Runs the regular post-close pipeline with `python daily_job.py --capture-type post_close`.
- `tail_capture`
  Runs the 14:30 intraday pipeline with an auto-generated `--snapshot-time` in `Asia/Shanghai`.
- `recompute`
  Rebuilds local outputs from existing cached data with `python daily_job.py --no-fetch`.
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

## Recommended Usage

If you want a cloud-based replacement for local automations:

1. Push this project to GitHub.
2. Open the Actions tab and enable workflows.
3. Use `workflow_dispatch` to test `tail_capture` and `full` manually.
4. After validation, rely on the default schedule.

## Important Note

Artifacts are stored on GitHub Actions and are not synced back to your local machine automatically. If you want the cloud run results to become the main data source for your local Streamlit dashboard later, add either:

- an explicit download-and-sync step on your local machine, or
- a follow-up workflow that commits generated report files back to a branch.
