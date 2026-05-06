from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.workflow_tasks import DEFAULT_WORKFLOW_TIMEZONE, run_workflow_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run workflow-friendly Top100 momentum tasks.")
    parser.add_argument(
        "--mode",
        required=True,
        help="Workflow mode: full, tail_capture, recompute, backtest, tests.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional markdown summary output path.",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional log directory path.",
    )
    parser.add_argument(
        "--snapshot-time",
        default=None,
        help="Optional snapshot timestamp override for tail_capture mode.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_WORKFLOW_TIMEZONE,
        help="Timezone used for auto-generated tail snapshot timestamps.",
    )
    parser.add_argument(
        "--force-refresh-prices",
        action="store_true",
        help="Force remote price refresh in full or tail_capture mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_workflow_mode(
        args.mode,
        force_refresh_prices=args.force_refresh_prices,
        snapshot_time=args.snapshot_time,
        timezone=args.timezone,
        summary_path=Path(args.summary_path) if args.summary_path else None,
        log_dir=Path(args.log_dir) if args.log_dir else None,
    )
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
