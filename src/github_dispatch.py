from __future__ import annotations

import os
from dataclasses import dataclass

import requests

from src.workflow_tasks import SUPPORTED_WORKFLOW_MODES, validate_workflow_mode

DEFAULT_GITHUB_API_VERSION = "2022-11-28"
DEFAULT_GITHUB_API_BASE = "https://api.github.com"
DEFAULT_REPOSITORY = "hzhipeng840-prog/top100_momentum_system"
DEFAULT_WORKFLOW_FILE = "top100_pipeline.yml"


@dataclass(frozen=True)
class WorkflowDispatchConfig:
    repository: str
    workflow_file: str
    ref: str
    mode: str
    force_refresh_prices: bool
    token: str
    api_base: str = DEFAULT_GITHUB_API_BASE
    api_version: str = DEFAULT_GITHUB_API_VERSION
    user_agent: str = "top100-momentum-dispatch/1.0"


def supported_dispatch_modes() -> tuple[str, ...]:
    return SUPPORTED_WORKFLOW_MODES


def build_workflow_dispatch_url(
    repository: str = DEFAULT_REPOSITORY,
    workflow_file: str = DEFAULT_WORKFLOW_FILE,
    api_base: str = DEFAULT_GITHUB_API_BASE,
) -> str:
    normalized_repo = str(repository or "").strip().strip("/")
    normalized_workflow = str(workflow_file or "").strip().lstrip("/")
    if not normalized_repo:
        raise ValueError("repository is required")
    if not normalized_workflow:
        raise ValueError("workflow_file is required")
    return f"{api_base.rstrip('/')}/repos/{normalized_repo}/actions/workflows/{normalized_workflow}/dispatches"


def build_workflow_dispatch_payload(
    *,
    ref: str = "main",
    mode: str = "full",
    force_refresh_prices: bool = False,
) -> dict[str, object]:
    normalized_mode = validate_workflow_mode(mode)
    return {
        "ref": str(ref or "main"),
        "inputs": {
            "mode": normalized_mode,
            "force_refresh_prices": bool(force_refresh_prices),
        },
    }


def build_workflow_dispatch_headers(
    token: str,
    *,
    api_version: str = DEFAULT_GITHUB_API_VERSION,
    user_agent: str = "top100-momentum-dispatch/1.0",
) -> dict[str, str]:
    resolved_token = str(token or "").strip()
    if not resolved_token:
        raise ValueError("token is required")
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {resolved_token}",
        "X-GitHub-Api-Version": str(api_version or DEFAULT_GITHUB_API_VERSION),
        "User-Agent": str(user_agent or "top100-momentum-dispatch/1.0"),
    }


def dispatch_workflow(config: WorkflowDispatchConfig, timeout_seconds: int = 30) -> dict[str, object]:
    url = build_workflow_dispatch_url(
        repository=config.repository,
        workflow_file=config.workflow_file,
        api_base=config.api_base,
    )
    payload = build_workflow_dispatch_payload(
        ref=config.ref,
        mode=config.mode,
        force_refresh_prices=config.force_refresh_prices,
    )
    headers = build_workflow_dispatch_headers(
        token=config.token,
        api_version=config.api_version,
        user_agent=config.user_agent,
    )

    response = requests.post(url, json=payload, headers=headers, timeout=timeout_seconds)
    response.raise_for_status()
    return {
        "status_code": response.status_code,
        "repository": config.repository,
        "workflow_file": config.workflow_file,
        "ref": config.ref,
        "mode": validate_workflow_mode(config.mode),
        "force_refresh_prices": bool(config.force_refresh_prices),
        "accepted": response.status_code in {200, 201, 202, 204},
    }


def config_from_env(
    *,
    mode: str,
    force_refresh_prices: bool = False,
    repository: str | None = None,
    workflow_file: str | None = None,
    ref: str | None = None,
    token_env: str = "GITHUB_TOKEN",
) -> WorkflowDispatchConfig:
    token = os.environ.get(token_env, "")
    return WorkflowDispatchConfig(
        repository=repository or os.environ.get("GITHUB_REPOSITORY", DEFAULT_REPOSITORY),
        workflow_file=workflow_file or os.environ.get("GITHUB_WORKFLOW_FILE", DEFAULT_WORKFLOW_FILE),
        ref=ref or os.environ.get("GITHUB_WORKFLOW_REF", "main"),
        mode=mode,
        force_refresh_prices=force_refresh_prices,
        token=token,
    )
