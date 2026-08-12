from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from src.backtest_service import run_backtest_service
from src.pipeline import run_pipeline
from src.settlement_repair import run_settlement_repair
from src.settings import load_settings
from src.strategy_profiles import available_strategy_versions, normalize_strategy_version


DEFAULT_RUN_MODE_TIMEZONE = "Asia/Shanghai"
PIPELINE_RUN_MODES = ("full", "morning_capture", "tail_capture", "recompute", "backtest", "nightly_reports", "settlement_repair")


@dataclass(frozen=True)
class PipelineModeConfig:
    mode: str
    native_fetch: bool
    capture_type: str | None
    snapshot_time: str | None


def validate_pipeline_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized not in PIPELINE_RUN_MODES:
        supported = ", ".join(PIPELINE_RUN_MODES)
        raise ValueError(f"Unsupported pipeline mode: {mode!r}. Supported modes: {supported}.")
    return normalized


def default_tail_snapshot_time(
    run_time: datetime | None = None,
    timezone: str = DEFAULT_RUN_MODE_TIMEZONE,
    hour: int = 14,
    minute: int = 30,
) -> str:
    zone = ZoneInfo(timezone)
    current = run_time.astimezone(zone) if run_time is not None else datetime.now(zone)
    snapshot = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return snapshot.strftime("%Y-%m-%d %H:%M:%S")


def default_morning_snapshot_time(
    run_time: datetime | None = None,
    timezone: str = DEFAULT_RUN_MODE_TIMEZONE,
    hour: int = 9,
    minute: int = 50,
) -> str:
    return default_tail_snapshot_time(run_time=run_time, timezone=timezone, hour=hour, minute=minute)


def resolve_pipeline_mode_config(
    mode: str,
    *,
    snapshot_time: str | None = None,
    timezone: str = DEFAULT_RUN_MODE_TIMEZONE,
) -> PipelineModeConfig:
    normalized = validate_pipeline_mode(mode)
    if normalized == "full":
        return PipelineModeConfig(mode=normalized, native_fetch=True, capture_type="post_close", snapshot_time=snapshot_time)
    if normalized == "morning_capture":
        return PipelineModeConfig(
            mode=normalized,
            native_fetch=True,
            capture_type="intraday_0950",
            snapshot_time=snapshot_time or default_morning_snapshot_time(timezone=timezone),
        )
    if normalized == "tail_capture":
        return PipelineModeConfig(
            mode=normalized,
            native_fetch=True,
            capture_type="intraday_1430",
            snapshot_time=snapshot_time or default_tail_snapshot_time(timezone=timezone),
        )
    if normalized == "recompute":
        return PipelineModeConfig(mode=normalized, native_fetch=False, capture_type=None, snapshot_time=None)
    if normalized == "nightly_reports":
        return PipelineModeConfig(mode=normalized, native_fetch=False, capture_type="post_close", snapshot_time=None)
    return PipelineModeConfig(mode=normalized, native_fetch=False, capture_type=None, snapshot_time=None)


def run_named_mode(
    mode: str,
    *,
    force_refresh_prices: bool = False,
    snapshot_time: str | None = None,
    timezone: str = DEFAULT_RUN_MODE_TIMEZONE,
) -> dict[str, object]:
    config = resolve_pipeline_mode_config(mode, snapshot_time=snapshot_time, timezone=timezone)
    if config.mode == "settlement_repair":
        return run_settlement_repair()
    if config.mode == "nightly_reports":
        initial_result = run_pipeline(
            native_fetch=False,
            capture_type=config.capture_type,
            snapshot_time=config.snapshot_time,
            force_refresh_prices=False,
            light_reports=True,
        )
        initial_freshness = initial_result.get("freshness", {}) if isinstance(initial_result.get("freshness"), dict) else {}
        initial_freshness_ok = bool(initial_freshness.get("is_fresh")) if "is_fresh" in initial_freshness else False
        repair_result: dict[str, object] = {
            "mode": "settlement_repair",
            "success": True,
            "skipped": True,
            "reason": "initial_freshness_ok",
        }
        if not initial_freshness_ok:
            repair_result = run_settlement_repair()
        if not initial_freshness_ok and not bool(repair_result.get("success")):
            initial_result["mode"] = config.mode
            initial_result["nightly_settlement_repair"] = repair_result
            initial_result["success"] = False
            return initial_result
        result = run_pipeline(
            native_fetch=False,
            capture_type=config.capture_type,
            snapshot_time=config.snapshot_time,
            force_refresh_prices=False,
            light_reports=False,
        )
        result["mode"] = config.mode
        result["nightly_initial_freshness"] = initial_freshness
        result["nightly_settlement_repair"] = repair_result
        freshness = result.get("freshness", {}) if isinstance(result.get("freshness"), dict) else {}
        freshness_ok = bool(freshness.get("is_fresh")) if "is_fresh" in freshness else False
        result["success"] = freshness_ok and (initial_freshness_ok or bool(repair_result.get("success")))
        return result
    if config.mode == "backtest":
        settings = load_settings()
        strategy_versions = available_strategy_versions(settings)
        results: dict[str, dict[str, object]] = {}
        for version in strategy_versions:
            normalized_version = normalize_strategy_version(version)
            service_result = run_backtest_service(strategy_version=normalized_version)
            results[normalized_version] = {
                "generated_at": service_result.generated_at,
                "summary_rows": len(service_result.summary_df),
                "rule_evaluation_rows": len(service_result.rule_evaluation_df),
            }
        return {
            "mode": config.mode,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "strategy_versions": strategy_versions,
            "backtests": results,
        }
    result = run_pipeline(
        native_fetch=config.native_fetch,
        capture_type=config.capture_type,
        snapshot_time=config.snapshot_time,
        force_refresh_prices=force_refresh_prices,
    )
    result["mode"] = config.mode
    return result
