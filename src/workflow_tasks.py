from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.paths import PROJECT_ROOT
from src.run_modes import DEFAULT_RUN_MODE_TIMEZONE, default_morning_snapshot_time, default_tail_snapshot_time


SUPPORTED_WORKFLOW_MODES = ("full", "morning_capture", "tail_capture", "recompute", "backtest", "nightly_reports", "settlement_repair", "tests")
DEFAULT_WORKFLOW_TIMEZONE = DEFAULT_RUN_MODE_TIMEZONE


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
    if run_time is None:
        return default_tail_snapshot_time(timezone=timezone, hour=hour, minute=minute)
    zone = ZoneInfo(timezone)
    current = run_time.astimezone(zone)
    snapshot_time = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return snapshot_time.strftime("%Y-%m-%d %H:%M:%S")


def build_morning_snapshot_time(
    run_time: datetime | None = None,
    timezone: str = DEFAULT_WORKFLOW_TIMEZONE,
    hour: int = 9,
    minute: int = 50,
) -> str:
    if run_time is None:
        return default_morning_snapshot_time(timezone=timezone, hour=hour, minute=minute)
    zone = ZoneInfo(timezone)
    current = run_time.astimezone(zone)
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

    if normalized == "tail_capture":
        resolved_snapshot_time = snapshot_time or build_tail_snapshot_time(timezone=timezone)
        command = [
            python_cmd,
            "daily_job.py",
            "--mode",
            normalized,
            "--snapshot-time",
            resolved_snapshot_time,
            "--timezone",
            timezone,
        ]
    elif normalized == "morning_capture":
        resolved_snapshot_time = snapshot_time or build_morning_snapshot_time(timezone=timezone)
        command = [
            python_cmd,
            "daily_job.py",
            "--mode",
            normalized,
            "--snapshot-time",
            resolved_snapshot_time,
            "--timezone",
            timezone,
        ]
    elif normalized in {"full", "recompute", "backtest", "nightly_reports", "settlement_repair"}:
        command = [python_cmd, "daily_job.py", "--mode", normalized, "--timezone", timezone]
    else:
        command = [python_cmd, "-m", "unittest", "discover", "-s", "tests", "-v"]

    if force_refresh_prices and normalized in {"full", "tail_capture", "morning_capture"}:
        command.append("--force-refresh-prices")
    return command


def _parse_pipeline_payload(output_text: str) -> dict[str, object] | None:
    text = str(output_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
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
        freshness_payload = payload.get("freshness", {}) if isinstance(payload.get("freshness"), dict) else {}
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
        if payload.get("deferred"):
            lines.append(f"- Deferred: `{payload.get('deferred_reason', '')}`")
            if payload.get("deferred_summary"):
                lines.append(f"- Deferred Summary: {payload.get('deferred_summary')}")
        if freshness_payload:
            lines.extend(
                [
                    f"- Freshness Status: `{freshness_payload.get('status', '')}`",
                    f"- Settlement: `{freshness_payload.get('settled_1d_row_count', 0)}/{freshness_payload.get('settlement_row_count', 0)}`",
                    f"- Freshness Summary: {freshness_payload.get('summary', '')}",
                ]
            )
        backtests_payload = payload.get("backtests", {}) if isinstance(payload.get("backtests"), dict) else {}
        if backtests_payload:
            lines.extend(["", "## Backtest Service", ""])
            for version, stats in backtests_payload.items():
                if not isinstance(stats, dict):
                    continue
                lines.append(
                    f"- `{version}`: summary `{stats.get('summary_rows', 0)}`, "
                    f"rule_eval `{stats.get('rule_evaluation_rows', 0)}`, generated `{stats.get('generated_at', '-')}`"
                )
        nightly_repair_payload = (
            payload.get("nightly_settlement_repair", {})
            if isinstance(payload.get("nightly_settlement_repair"), dict)
            else {}
        )
        if nightly_repair_payload:
            lines.extend(
                [
                    "",
                    "## Nightly Settlement Repair",
                    "",
                    f"- Status: {'success' if nightly_repair_payload.get('success') else 'failed'}",
                    f"- Target Dates: `{', '.join(nightly_repair_payload.get('target_dates', []))}`",
                    f"- Repair Codes: `{nightly_repair_payload.get('repair_code_count', 0)}`",
                    f"- Remaining Stale Versions: `{', '.join(sorted((nightly_repair_payload.get('remaining_stale_codes_by_version') or {}).keys()))}`",
                ]
            )
            price_repair = nightly_repair_payload.get("price_repair", [])
            if isinstance(price_repair, list):
                for item in price_repair:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"- Price Repair `{item.get('target_date', '')}`: "
                        f"{item.get('repaired_count', 0)}/{item.get('requested_count', 0)} repaired, "
                        f"{item.get('remaining_count', 0)} remaining"
                    )
                    if item.get("stopped_early"):
                        lines.append(f"- Price Repair Stop `{item.get('target_date', '')}`: {item.get('stop_reason', '')}")
            repair_freshness = (
                nightly_repair_payload.get("freshness", {})
                if isinstance(nightly_repair_payload.get("freshness"), dict)
                else {}
            )
            if repair_freshness:
                lines.append(f"- Final Freshness: `{repair_freshness.get('status', '')}` {repair_freshness.get('summary', '')}")
            historical_backfill = (
                nightly_repair_payload.get("historical_backfill", {})
                if isinstance(nightly_repair_payload.get("historical_backfill"), dict)
                else {}
            )
            if historical_backfill.get("enabled"):
                lines.append(
                    f"- Historical Backfill: batch `{historical_backfill.get('batch_code_count', 0)}`, "
                    f"repaired `{historical_backfill.get('repaired_count', 0)}`, "
                    f"remaining backlog `{historical_backfill.get('remaining_backlog_count', 0)}`"
                )

    repair_result = result.get("repair") if isinstance(result.get("repair"), dict) else None
    if repair_result:
        repair_payload = repair_result.get("payload") if isinstance(repair_result.get("payload"), dict) else {}
        lines.extend(
            [
                "",
                "## Settlement Repair",
                "",
                f"- Status: {'success' if repair_result.get('success') else 'failed'}",
                f"- Return Code: `{repair_result.get('returncode', '')}`",
                f"- Command: `{repair_result.get('command_display', '')}`",
                f"- Log File: `{repair_result.get('log_path', '')}`",
            ]
        )
        if repair_payload:
            lines.extend(
                [
                    f"- Target Dates: `{', '.join(repair_payload.get('target_dates', []))}`",
                    f"- Repair Codes: `{repair_payload.get('repair_code_count', 0)}`",
                    f"- Remaining Stale Versions: `{', '.join(sorted((repair_payload.get('remaining_stale_codes_by_version') or {}).keys()))}`",
                ]
            )
            price_repair = repair_payload.get("price_repair", [])
            if isinstance(price_repair, list):
                for item in price_repair:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"- Price Repair `{item.get('target_date', '')}`: "
                        f"{item.get('repaired_count', 0)}/{item.get('requested_count', 0)} repaired, "
                        f"{item.get('remaining_count', 0)} remaining"
                    )
                    if item.get("stopped_early"):
                        lines.append(f"- Price Repair Stop `{item.get('target_date', '')}`: {item.get('stop_reason', '')}")
            repair_freshness = repair_payload.get("freshness", {}) if isinstance(repair_payload.get("freshness"), dict) else {}
            if repair_freshness:
                lines.append(f"- Final Freshness: `{repair_freshness.get('status', '')}` {repair_freshness.get('summary', '')}")

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

    resolved_snapshot_time = snapshot_time if normalized in {"tail_capture", "morning_capture"} else None
    if normalized == "tail_capture" and not resolved_snapshot_time:
        resolved_snapshot_time = build_tail_snapshot_time(timezone=timezone)
    if normalized == "morning_capture" and not resolved_snapshot_time:
        resolved_snapshot_time = build_morning_snapshot_time(timezone=timezone)

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
        if normalized in {"settlement_repair", "nightly_reports"} and "success" in payload:
            result["success"] = completed.returncode == 0 and bool(payload.get("success"))

    data_payload = payload.get("data", {}) if isinstance(payload, dict) and isinstance(payload.get("data"), dict) else {}
    if normalized == "full" and completed.returncode == 0 and data_payload.get("status") == "stale_settlement":
        repair_command = build_workflow_command(
            "settlement_repair",
            timezone=timezone,
            python_executable=command[0],
        )
        repair_started_at = datetime.now().astimezone()
        repair_log_path = log_root / f"settlement_repair_{timestamp}.log"
        repair_completed = subprocess.run(
            repair_command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        repair_finished_at = datetime.now().astimezone()
        repair_log_sections = [
            f"Command: {subprocess.list2cmdline(repair_command)}",
            f"Started At: {repair_started_at.isoformat(timespec='seconds')}",
            f"Finished At: {repair_finished_at.isoformat(timespec='seconds')}",
            f"Return Code: {repair_completed.returncode}",
            "",
            "[STDOUT]",
            repair_completed.stdout.rstrip(),
            "",
            "[STDERR]",
            repair_completed.stderr.rstrip(),
            "",
        ]
        repair_log_path.write_text("\n".join(repair_log_sections), encoding="utf-8")

        repair_payload = _parse_pipeline_payload(repair_completed.stdout)
        repair_success = repair_completed.returncode == 0
        if isinstance(repair_payload, dict) and "success" in repair_payload:
            repair_success = repair_success and bool(repair_payload.get("success"))
        result["repair"] = {
            "mode": "settlement_repair",
            "success": repair_success,
            "returncode": repair_completed.returncode,
            "command": repair_command,
            "command_display": subprocess.list2cmdline(repair_command),
            "log_path": str(repair_log_path),
            "stdout": repair_completed.stdout,
            "stderr": repair_completed.stderr,
            "started_at": repair_started_at.isoformat(timespec="seconds"),
            "finished_at": repair_finished_at.isoformat(timespec="seconds"),
            "payload": repair_payload,
        }
        result["success"] = bool(result.get("success")) and repair_success
        result["finished_at"] = repair_finished_at.isoformat(timespec="seconds")

    if summary_path is not None:
        write_workflow_summary(summary_path, result)

    return result
