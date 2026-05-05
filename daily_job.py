from __future__ import annotations

import argparse
import json

from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Top100 momentum research pipeline.")
    parser.add_argument("--no-fetch", action="store_true", help="Only recompute existing local data without fetching.")
    parser.add_argument("--capture-type", default=None, help="Snapshot label, such as post_close, intraday_0935, intraday_1430.")
    parser.add_argument("--snapshot-time", default=None, help="Snapshot timestamp, for example 2026-04-20 09:35:00.")
    parser.add_argument("--force-refresh-prices", action="store_true", help="Force remote refresh of stock price caches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        native_fetch=not args.no_fetch,
        capture_type=args.capture_type,
        snapshot_time=args.snapshot_time,
        force_refresh_prices=args.force_refresh_prices,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
