from __future__ import annotations

import argparse
import json

from src.github_dispatch import (
    DEFAULT_REPOSITORY,
    DEFAULT_WORKFLOW_FILE,
    config_from_env,
    dispatch_workflow,
    supported_dispatch_modes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger the GitHub workflow_dispatch API for this repository.")
    parser.add_argument("--mode", required=True, choices=supported_dispatch_modes(), help="Workflow mode to dispatch.")
    parser.add_argument("--force-refresh-prices", action="store_true", help="Force remote price refresh for full or tail_capture.")
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="GitHub repository in owner/name form.")
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE, help="Workflow file name under .github/workflows/.")
    parser.add_argument("--ref", default="main", help="Git ref to dispatch on.")
    parser.add_argument("--token-env", default="GITHUB_TOKEN", help="Environment variable name that holds the GitHub token.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = config_from_env(
        mode=args.mode,
        force_refresh_prices=args.force_refresh_prices,
        repository=args.repository,
        workflow_file=args.workflow_file,
        ref=args.ref,
        token_env=args.token_env,
    )
    result = dispatch_workflow(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
