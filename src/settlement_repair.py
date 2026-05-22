from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from src.freshness import build_data_freshness_report
from src.followups import build_followups, save_followups
from src.native_fetcher import warm_stock_price_cache
from src.paths import MARKET_REGIME_CSV, RAW_STOCK_PRICE_DIR, followups_csv_for, signals_csv_for
from src.pipeline import _optional_positive_int, _stale_settlement_codes
from src.reports import build_reports
from src.settings import load_settings
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, available_strategy_versions, normalize_strategy_version
from src.utils import normalize_code, read_csv_safely


DEFAULT_REPAIR_MAX_ATTEMPTS = 3
DEFAULT_REPAIR_MAX_WORKERS = 1
DEFAULT_REPAIR_SLEEP_SECONDS = 3.0


def _normalize_date_text(value: object) -> str:
    parsed = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).normalize().strftime("%Y-%m-%d")


def _price_cache_reaches_date(code: object, target_date: object) -> bool:
    normalized_code = normalize_code(code)
    target_date_text = _normalize_date_text(target_date)
    if not normalized_code or not target_date_text:
        return False

    price_df = read_csv_safely(RAW_STOCK_PRICE_DIR / f"{normalized_code}.csv")
    if price_df.empty or "date" not in price_df.columns:
        return False

    dates = pd.to_datetime(price_df["date"], errors="coerce").dropna()
    if dates.empty:
        return False
    return pd.Timestamp(dates.max()).normalize() >= pd.Timestamp(target_date_text)


def _missing_price_codes(codes: set[str], target_date: object) -> list[str]:
    return sorted(code for code in codes if not _price_cache_reaches_date(code, target_date))


def _settlement_codes(followup_df: pd.DataFrame, settlement_date: object) -> set[str]:
    settlement_date_text = _normalize_date_text(settlement_date)
    if followup_df.empty or "signal_date" not in followup_df.columns or "code" not in followup_df.columns or not settlement_date_text:
        return set()

    working = followup_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rows = working[working["signal_date"].eq(settlement_date_text)].copy()
    codes = set(rows["code"].dropna().astype(str).map(normalize_code))
    return {code for code in codes if code}


def repair_price_caches_for_target(
    codes: set[str],
    target_date: object,
    *,
    max_attempts: int = DEFAULT_REPAIR_MAX_ATTEMPTS,
    max_workers: int = DEFAULT_REPAIR_MAX_WORKERS,
    retry_sleep_seconds: float = DEFAULT_REPAIR_SLEEP_SECONDS,
) -> dict[str, object]:
    target_date_text = _normalize_date_text(target_date)
    normalized_codes = {normalize_code(code) for code in codes if normalize_code(code)}
    attempt_count = max(1, int(max_attempts))
    attempts: list[dict[str, object]] = []
    remaining = _missing_price_codes(normalized_codes, target_date_text)

    for attempt in range(1, attempt_count + 1):
        if not remaining:
            break
        stats = warm_stock_price_cache(
            remaining,
            force_refresh=True,
            max_workers=max(1, int(max_workers)),
        )
        remaining = _missing_price_codes(normalized_codes, target_date_text)
        attempts.append(
            {
                "attempt": attempt,
                "requested": stats.get("requested", 0),
                "remote": stats.get("remote", 0),
                "cache": stats.get("cache", 0),
                "stale_cache": stats.get("stale_cache", 0),
                "missing": stats.get("missing", 0),
                "remaining": len(remaining),
            }
        )
        if remaining and attempt < attempt_count and retry_sleep_seconds > 0:
            time.sleep(float(retry_sleep_seconds))

    repaired = sorted(normalized_codes - set(remaining))
    return {
        "target_date": target_date_text,
        "requested_codes": sorted(normalized_codes),
        "requested_count": len(normalized_codes),
        "repaired_codes": repaired,
        "repaired_count": len(repaired),
        "remaining_codes": remaining,
        "remaining_count": len(remaining),
        "attempts": attempts,
        "success": not remaining,
    }


def _repair_targets(
    strategy_versions: list[str],
    market_regime_df: pd.DataFrame,
    *,
    min_settlement_ratio: float,
) -> tuple[dict[str, dict[str, object]], dict[str, set[str]], dict[str, str]]:
    freshness_by_version: dict[str, dict[str, object]] = {}
    stale_codes_by_version: dict[str, set[str]] = {}
    target_date_by_version: dict[str, str] = {}

    for strategy_version in strategy_versions:
        normalized_version = normalize_strategy_version(strategy_version)
        followup_df = read_csv_safely(followups_csv_for(normalized_version))
        freshness = build_data_freshness_report(
            followup_df,
            market_regime_df,
            min_settlement_ratio=min_settlement_ratio,
        )
        freshness_by_version[normalized_version] = freshness
        settled_ratio = freshness.get("settled_1d_ratio")
        settlement_row_count = int(freshness.get("settlement_row_count") or 0)
        if settlement_row_count <= 0 or settled_ratio is None:
            continue
        if float(settled_ratio) >= float(min_settlement_ratio):
            continue
        settlement_date = freshness.get("settlement_date")
        stale_codes = _stale_settlement_codes(followup_df, settlement_date)
        repair_codes = _settlement_codes(followup_df, settlement_date)
        if not stale_codes or not repair_codes:
            continue
        stale_codes_by_version[normalized_version] = repair_codes
        target_date_by_version[normalized_version] = str(freshness.get("expected_market_date") or "")

    return freshness_by_version, stale_codes_by_version, target_date_by_version


def run_settlement_repair(
    *,
    max_attempts: int = DEFAULT_REPAIR_MAX_ATTEMPTS,
    max_workers: int = DEFAULT_REPAIR_MAX_WORKERS,
    retry_sleep_seconds: float = DEFAULT_REPAIR_SLEEP_SECONDS,
) -> dict[str, object]:
    settings = load_settings()
    strategy_versions = available_strategy_versions(settings)
    default_strategy_version = normalize_strategy_version(settings.get("default_strategy_version", DEFAULT_STRATEGY_VERSION))
    followup_days = list(settings.get("followup_days", [1, 3, 5, 10]))
    latest_push_limit = _optional_positive_int(settings.get("latest_push_limit"))
    strong_threshold = float(settings.get("strong_return_threshold_pct", 15))
    freshness_min_ratio = float(settings.get("settlement_freshness_min_ratio", 0.95))
    market_regime_df = read_csv_safely(MARKET_REGIME_CSV)

    started_at = datetime.now().isoformat(timespec="seconds")
    before_freshness, stale_codes_by_version, target_date_by_version = _repair_targets(
        strategy_versions,
        market_regime_df,
        min_settlement_ratio=freshness_min_ratio,
    )
    repair_codes: set[str] = set()
    target_dates: set[str] = set()
    for version, codes in stale_codes_by_version.items():
        repair_codes.update(codes)
        target_date = target_date_by_version.get(version, "")
        if target_date:
            target_dates.add(target_date)

    price_repair_results: list[dict[str, object]] = []
    for target_date in sorted(target_dates):
        price_repair_results.append(
            repair_price_caches_for_target(
                repair_codes,
                target_date,
                max_attempts=max_attempts,
                max_workers=max_workers,
                retry_sleep_seconds=retry_sleep_seconds,
            )
        )

    report_results: dict[str, dict[str, object]] = {}
    repaired_versions = sorted(stale_codes_by_version)
    if repaired_versions and all(result.get("success") for result in price_repair_results):
        for strategy_version in repaired_versions:
            signal_df = read_csv_safely(signals_csv_for(strategy_version))
            followup_df = build_followups(signal_df, days=followup_days, strategy_version=strategy_version)
            save_followups(followup_df, strategy_version=strategy_version)
            report_results[strategy_version] = build_reports(
                signal_df=signal_df,
                followup_df=followup_df,
                latest_push_limit=latest_push_limit,
                strong_return_threshold_pct=strong_threshold,
                strategy_version=strategy_version,
                light_mode=True,
            )

    after_freshness, after_stale_codes_by_version, _target_date_by_version = _repair_targets(
        strategy_versions,
        market_regime_df,
        min_settlement_ratio=freshness_min_ratio,
    )
    default_freshness = after_freshness.get(default_strategy_version, {})
    success = bool(default_freshness.get("is_fresh")) and not after_stale_codes_by_version

    return {
        "mode": "settlement_repair",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "success": success,
        "strategy_versions": strategy_versions,
        "default_strategy_version": default_strategy_version,
        "repair_versions": repaired_versions,
        "repair_codes": sorted(repair_codes),
        "repair_code_count": len(repair_codes),
        "target_dates": sorted(target_dates),
        "price_repair": price_repair_results,
        "reports": report_results,
        "freshness_before": before_freshness,
        "freshness_after": after_freshness,
        "remaining_stale_codes_by_version": {
            version: sorted(codes)
            for version, codes in after_stale_codes_by_version.items()
        },
        "freshness": default_freshness,
    }
