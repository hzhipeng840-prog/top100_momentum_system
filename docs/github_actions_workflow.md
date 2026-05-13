# GitHub Actions and External Cron

This project now uses GitHub Actions for:

- `push -> tests`
- manual `workflow_dispatch`
- cloud execution after an external scheduler calls `workflow_dispatch`

It no longer relies on GitHub's own `schedule` trigger for time-sensitive A-share capture windows.

## Supported Workflow Modes

- `full`
  Runs the post-close pipeline with `python daily_job.py --mode full`.
- `morning_capture`
  Runs the 9:50 intraday pipeline with an explicit Shanghai snapshot timestamp.
- `tail_capture`
  Runs the 14:30 intraday pipeline with an explicit Shanghai snapshot timestamp.
- `recompute`
  Rebuilds local outputs from existing cached data without fetching new market data. This mode is local-only in spirit: the workflow still runs in GitHub and uploads artifacts, but it does not sync generated outputs back to `main`.
- `backtest`
  Rebuilds formal backtest summary outputs from existing `signals` and `followups`.
- `tests`
  Runs `python -m unittest discover -s tests -v`.

## Why External Cron

GitHub's `schedule` trigger is acceptable for low-precision daily jobs, but it is not reliable enough for:

- 14:30 A-share tail capture
- 17:10 post-close jobs that should stay close to the market window

This repository keeps GitHub Actions as the execution environment, but moves the timing responsibility to an external scheduler.

## Workflow Dispatch Endpoint

Workflow file:

- `.github/workflows/top100_pipeline.yml`

Dispatch target:

- `https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches`

## Token Requirements

Use one of these:

- Fine-grained personal access token with repository `Actions: Read and write`
- Classic personal access token with `repo` scope

Store it outside the repository as an environment variable:

- `GITHUB_TOKEN`

## Dispatch Script

This repository includes a ready-to-use helper:

- `dispatch_workflow.py`

Examples:

```bash
python dispatch_workflow.py --mode morning_capture
python dispatch_workflow.py --mode tail_capture
python dispatch_workflow.py --mode full
python dispatch_workflow.py --mode backtest
```

Optional flags:

- `--force-refresh-prices`
- `--repository`
- `--workflow-file`
- `--ref`
- `--token-env`

## Direct API Example

Morning capture:

```bash
curl -L -X POST ^
  -H "Accept: application/vnd.github+json" ^
  -H "Authorization: Bearer %GITHUB_TOKEN%" ^
  -H "X-GitHub-Api-Version: 2022-11-28" ^
  https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches ^
  -d "{\"ref\":\"main\",\"inputs\":{\"mode\":\"morning_capture\",\"force_refresh_prices\":false}}"
```

Tail capture:

```bash
curl -L -X POST ^
  -H "Accept: application/vnd.github+json" ^
  -H "Authorization: Bearer %GITHUB_TOKEN%" ^
  -H "X-GitHub-Api-Version: 2022-11-28" ^
  https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches ^
  -d "{\"ref\":\"main\",\"inputs\":{\"mode\":\"tail_capture\",\"force_refresh_prices\":false}}"
```

Full run:

```bash
curl -L -X POST ^
  -H "Accept: application/vnd.github+json" ^
  -H "Authorization: Bearer %GITHUB_TOKEN%" ^
  -H "X-GitHub-Api-Version: 2022-11-28" ^
  https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches ^
  -d "{\"ref\":\"main\",\"inputs\":{\"mode\":\"full\",\"force_refresh_prices\":false}}"
```

## Example External Schedules

### Linux or VPS cron

```cron
50 9 * * 1-5 cd /opt/top100_momentum_system && /usr/bin/python3 dispatch_workflow.py --mode morning_capture
30 14 * * 1-5 cd /opt/top100_momentum_system && /usr/bin/python3 dispatch_workflow.py --mode tail_capture
10 17 * * 1-5 cd /opt/top100_momentum_system && /usr/bin/python3 dispatch_workflow.py --mode full
```

All times above should be interpreted in the machine's local timezone. If the server is not in China, align the cron expression to `Asia/Shanghai`.

### Third-party cron service

If the service supports custom headers and JSON bodies, call the GitHub REST endpoint directly with:

- `Authorization: Bearer <token>`
- `Accept: application/vnd.github+json`
- `X-GitHub-Api-Version: 2022-11-28`

Payload examples are shown above.

### Recommended zero-manual setup: cron-job.org

`cron-job.org` fits this repository well because it supports:

- custom headers
- `POST` requests with a JSON body
- execution history with response details
- failure email notifications

Recommended setup:

#### Job 1: 9:50 morning capture

- Name: `top100 morning_capture`
- URL: `https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches`
- Method: `POST`
- Schedule: weekdays at `09:50` in `Asia/Shanghai`
- Headers:
  - `Accept: application/vnd.github+json`
  - `Authorization: Bearer <YOUR_GITHUB_TOKEN>`
  - `X-GitHub-Api-Version: 2022-11-28`
- Body:

```json
{
  "ref": "main",
  "inputs": {
    "mode": "morning_capture",
    "force_refresh_prices": false
  }
}
```

#### Job 2: 14:30 tail capture

- Name: `top100 tail_capture`
- URL: `https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches`
- Method: `POST`
- Schedule: weekdays at `14:30` in `Asia/Shanghai`
- Headers:
  - `Accept: application/vnd.github+json`
  - `Authorization: Bearer <YOUR_GITHUB_TOKEN>`
  - `X-GitHub-Api-Version: 2022-11-28`
- Body:

```json
{
  "ref": "main",
  "inputs": {
    "mode": "tail_capture",
    "force_refresh_prices": false
  }
}
```

#### Job 3: 17:10 post-close full run

- Name: `top100 full`
- URL: `https://api.github.com/repos/hzhipeng840-prog/top100_momentum_system/actions/workflows/top100_pipeline.yml/dispatches`
- Method: `POST`
- Schedule: weekdays at `17:10` in `Asia/Shanghai`
- Headers:
  - `Accept: application/vnd.github+json`
  - `Authorization: Bearer <YOUR_GITHUB_TOKEN>`
  - `X-GitHub-Api-Version: 2022-11-28`
- Body:

```json
{
  "ref": "main",
  "inputs": {
    "mode": "full",
    "force_refresh_prices": false
  }
}
```

#### What you need to check day to day

After the two jobs are created, the intended operating model is:

- you do not trigger them manually
- you do not confirm them every day
- you only look when one of these happens:
  - `cron-job.org` sends a failure email
  - your local dashboard sync still shows an old sample date
  - GitHub Actions shows a failed `workflow_dispatch` run

Recommended safety switches in `cron-job.org`:

- enable failure email notifications
- keep execution history enabled
- keep the response body visible so you can inspect the GitHub API response

If you want a quick health check without opening GitHub every day, the simplest routine is:

1. open the local dashboard
2. click `同步最新云端结果`
3. confirm the latest sample date updated as expected

That is enough for normal use.

## What Gets Synced Back

For `workflow_dispatch` runs in these modes:

- `full`
- `morning_capture`
- `tail_capture`
- `backtest`

the workflow commits generated outputs back to `main`, including:

- `data/raw/popularity_top100.csv`
- `data/raw/intraday_snapshots.csv`
- `data/processed/daily_features.csv`
- versioned `signals`, `followups`, and `fast_strategy_history`
- `market_regime.csv`
- `data/reports/*.csv`

It does not sync outputs for `tests`.

`recompute` is intentionally excluded from sync-back because it only recalculates from whatever caches already exist in the runner. That makes it useful for diagnostics and local-style experiments, but not safe to publish as the repository's formal latest result.

## Recommended Operating Model

Use GitHub Actions for compute, and an external scheduler for timing:

1. External cron triggers `workflow_dispatch`
2. GitHub Actions executes the job in the cloud
3. Generated outputs sync back to `main`
4. Your local dashboard pulls the latest results with `git pull`
