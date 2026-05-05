from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.paths import PROJECT_ROOT


SUPPORTED_WORKFLOW_MODES = ("full", "tail_capture", "recompute", "tests")
DEFAULT_WORKFLOW_TIMEZONE = "Asia/Shanghai"


def validate_workflow_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in SUPPORTED_WORKFLOW_MODES:
        supported = ", ".join(SUPPORTED_WORKFLOW_MODES)
        raise ValueError(f"Unsupported workflow mode: {mode!r}. Supported modes: {supported}.")
    return normalized


def build_tail_snapshot_time(
    run_time: datetime | None = None,
    timezone: str = DEFAULT_WORKFLOW_TIMEZONE,
    hour: int = 14,
    minute: int = 30,
) -> str:
    zone = ZoneInfo(timezone)
    current = run_time.astimezone(zone) if run_time is not None else datetime.now(zone)
    snapshot_time = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return snapshot_time.strftime("%Y-%m-%d %H:%M:%S")


def build_workflow_command(
    mode: str,
    *,
    force_refresh_prices: bool = False,
    snapshot_time: str | None = None,
    timezone: str = DEFAULT_WORKFLOW_TIMEZONE,
    python_executable: str | None = None,
) -> list[str]:
    normalized = validate_workflow_mode(mode)
    python_cmd = python_executable or sys.executable

    if normalized == "full":
        command = [python_cmd, "daily_job.py", "--capture-type", "post_close"]
    elif normalized == "tail_capture":
        resolved_snapshot_time = snapshot_time or build_tail_snapshot_time(timezone=timezone)
        command = [
            python_cmd,
            "daily_job.py",
            "--capture-type",
            "intraday_1430",
            "--snapshot-time",
            resolved_snapshot_time,
        ]
    elif normalized == "recompute":
        command = [python_cmd, "daily_job.py", "--no-fetch"]
    else:
        command = [python_cmd, "-m", "unittest", "discover", "-s", "tests", "-v"]

    if force_refresh_prices and normalized in {"full", "tail_capture"}:
        command.append("--force-refresh-prices")
    return command


def _parse_pipeline_payload(output_text: str) -> dict[str, object] | None:
    text = str(output_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def write_workflow_summary(summary_path: Path, result: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    payload = result.get("payload") if isinstance(result.get("payload"), dict) else None
    lines = [
        "# Top100 Workflow Summary",
        "",
        f"- Status: {'success' if result.get('success') else 'failed'}",
        f"- Mode: `{result.get('mode', '')}`",
        f"- Started At: `{result.get('started_at', '')}`",
        f"- Finished At: `{result.get('finished_at', '')}`",
        f"- Return Code: `{result.get('returncode', '')}`",
        f"- Command: `{result.get('command_display', '')}`",
        f"- Log File: `{result.get('log_path', '')}`",
    ]

    if result.get("snapshot_time"):
        lines.append(f"- Snapshot Time: `{result.get('snapshot_time')}`")

    if payload:
        data_payload = payload.get("data", {}) if isinstance(payload.get("data"), dict) else {}
        features_payload = payload.get("features", {}) if isinstance(payload.get("features"), dict) else {}
        signals_payload = payload.get("signals", {}) if isinstance(payload.get("signals"), dict) else {}
        followups_payload = payload.get("followups", {}) if isinstance(payload.get("followups"), dict) else {}
        lines.extend(
            [
                "",
                "## Pipeline Result",
                "",
                f"- Data Status: `{data_payload.get('status', '')}`",
                f"- Strategy Versions: `{', '.join(payload.get('strategy_versions', []))}`",
                f"- Default Strategy Version: `{payload.get('default_strategy_version', '')}`",
                f"- Feature Rows: `{features_payload.get('rows', 0)}`",
                f"- Signal Rows: `{signals_payload.get('rows', 0)}`",
                f"- Pushed Rows: `{signals_payload.get('pushed_rows', 0)}`",
                f"- Followup Rows: `{followups_payload.get('rows', 0)}`",
            ]
        )

    if result.get("stderr"):
        lines.extend(["", "## STDERR", "", "```text", str(result["stderr"]).strip(), "```"])

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_workflow_mode(
    mode: str,
    *,
    force_refresh_prices: bool = False,
    snapshot_time: str | None = None,
    timezone: str = DEFAULT_WORKFLOW_TIMEZONE,
    summary_path: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, object]:
    normalized = validate_workflow_mode(mode)
    command = build_workflow_command(
        normalized,
        force_refresh_prices=force_refresh_prices,
        snapshot_time=snapshot_time,
        timezone=timezone,
    )
    log_root = log_dir or (PROJECT_ROOT / "workflow_artifacts" / "logs")
    log_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    log_path = log_root / f"{normalized}_{timestamp}.log"

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    finished_at = datetime.now().astimezone()
    log_sections = [
        f"Command: {subprocess.list2cmdline(command)}",
        f"Started At: {started_at.isoformat(timespec='seconds')}",
        f"Finished At: {finished_at.isoformat(timespec='seconds')}",
        f"Return Code: {completed.returncode}",
        "",
        "[STDOUT]",
        completed.stdout.rstrip(),
        "",
        "[STDERR]",
        completed.stderr.rstrip(),
        "",
    ]
    log_path.write_text("\n".join(log_sections), encoding="utf-8")

    resolved_snapshot_time = snapshot_time if normalized == "tail_capture" else None
    if normalized == "tail_capture" and not resolved_snapshot_time:
        resolved_snapshot_time = build_tail_snapshot_time(timezone=timezone)

    result: dict[str, object] = {
        "mode": normalized,
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "command_display": subprocess.list2cmdline(command),
        "log_path": str(log_path),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "snapshot_time": resolved_snapshot_time,
    }

    payload = _parse_pipeline_payload(completed.stdout)
    if payload is not None:
        result["payload"] = payload

    if summary_path is not None:
        write_workflow_summary(summary_path, result)

    return result
