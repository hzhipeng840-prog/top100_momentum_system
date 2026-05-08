from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.features import build_daily_features, load_popularity, save_daily_features
from src.followups import build_followups, save_followups
from src.intraday_fetcher import warm_intraday_cache
from src.market_regime import build_market_regime, save_market_regime, warm_market_index_cache
from src.native_fetcher import run_native_fetch, warm_stock_price_cache
from src.paths import FEATURES_CSV, ensure_layout, fast_strategy_history_csv_for, followups_csv_for
from src.reports import build_reports
from src.settings import load_settings
from src.signals import build_signals, save_signals
from src.strategy_profiles import (
    DEFAULT_STRATEGY_VERSION,
    available_strategy_versions,
    normalize_strategy_version,
    strategy_version_for_capture_type,
)
from src.trading_calendar import latest_expected_market_date, should_skip_market_fetch
from src.utils import normalize_code, read_csv_safely


def _optional_positive_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "0", "all", "none", "null"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _recent_signal_codes(signal_df: pd.DataFrame, lookback_dates: int) -> set[str]:
    if signal_df.empty or lookback_dates <= 0 or not {"signal_date", "code"}.issubset(signal_df.columns):
        return set()

    working = signal_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    recent_dates = sorted(date for date in working["signal_date"].dropna().astype(str).unique() if date)
    if not recent_dates:
        return set()

    selected_dates = set(recent_dates[-lookback_dates:])
    recent = working[working["signal_date"].isin(selected_dates)]
    return set(recent["code"].dropna().astype(str).map(normalize_code))


def _unfinished_followup_codes(
    signal_df: pd.DataFrame,
    max_followup_days: int,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> set[str]:
    if max_followup_days <= 0:
        return set()

    strategy_version = normalize_strategy_version(strategy_version)
    followup_df = read_csv_safely(followups_csv_for(strategy_version))
    if followup_df.empty or "code" not in followup_df.columns:
        return _recent_signal_codes(signal_df, max_followup_days)

    working = followup_df.copy()
    working["code"] = working["code"].dropna().astype(str).map(normalize_code)
    settled_column = f"settled_{max_followup_days}d"
    if settled_column in working.columns:
        unsettled = working[~_truthy_mask(working[settled_column])].copy()
    elif "observed_days" in working.columns:
        observed = pd.to_numeric(working["observed_days"], errors="coerce").fillna(0)
        unsettled = working[observed < max_followup_days].copy()
    else:
        return _recent_signal_codes(signal_df, max_followup_days)

    codes = set(unsettled["code"].dropna().astype(str).map(normalize_code))
    return {code for code in codes if code}


def _followup_refresh_codes(
    signal_df: pd.DataFrame,
    followup_days: list[int] | None = None,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> list[str]:
    strategy_version = normalize_strategy_version(strategy_version)
    codes: set[str] = set()
    history_df = read_csv_safely(fast_strategy_history_csv_for(strategy_version))
    if not history_df.empty and "code" in history_df.columns:
        codes.update(history_df["code"].dropna().astype(str).map(normalize_code))
    if not signal_df.empty and {"code", "is_pushed"}.issubset(signal_df.columns):
        pushed = signal_df[_truthy_mask(signal_df["is_pushed"])].copy()
        codes.update(pushed["code"].dropna().astype(str).map(normalize_code))
    max_followup_days = max((int(day) for day in (followup_days or []) if int(day) > 0), default=0)
    codes.update(_unfinished_followup_codes(signal_df, max_followup_days, strategy_version=strategy_version))
    return sorted(code for code in codes if code)


def _latest_signal_slice(signal_df: pd.DataFrame) -> pd.DataFrame:
    if signal_df.empty or "signal_date" not in signal_df.columns:
        return pd.DataFrame()

    working = signal_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working = working.dropna(subset=["signal_date"]).copy()
    if working.empty:
        return pd.DataFrame()

    latest_date = working["signal_date"].max()
    latest = working[working["signal_date"].astype(str).eq(str(latest_date))].copy()
    if latest.empty:
        return latest

    if "capture_type" in latest.columns:
        capture_series = latest["capture_type"].fillna("").astype(str)
        if capture_series.eq("post_close").any():
            latest = latest[capture_series.eq("post_close")].copy()

    if "snapshot_time" in latest.columns:
        latest["snapshot_time"] = latest["snapshot_time"].fillna("").astype(str)
        latest_snapshot = latest["snapshot_time"][latest["snapshot_time"].ne("")].max()
        if latest_snapshot:
            latest = latest[latest["snapshot_time"].eq(latest_snapshot)].copy()
    return latest


def _intraday_refresh_codes(
    signal_df: pd.DataFrame,
    pushed_only: bool = True,
    limit: int | None = None,
) -> list[str]:
    latest = _latest_signal_slice(signal_df)
    if latest.empty or "code" not in latest.columns:
        return []

    focus = latest.copy()
    if pushed_only:
        if "is_pushed" in focus.columns:
            pushed = focus[_truthy_mask(focus["is_pushed"])].copy()
            if not pushed.empty:
                focus = pushed
        elif "push_level" in focus.columns:
            pushed = focus[focus["push_level"].fillna("").astype(str).ne("不推送")].copy()
            if not pushed.empty:
                focus = pushed

    sort_by: list[str] = []
    ascending: list[bool] = []
    if "emotion_score" in focus.columns:
        sort_by.append("emotion_score")
        ascending.append(False)
    if "rank" in focus.columns:
        sort_by.append("rank")
        ascending.append(True)
    if sort_by:
        focus = focus.sort_values(sort_by, ascending=ascending, na_position="last")

    ordered_codes: list[str] = []
    seen: set[str] = set()
    for code in focus["code"].dropna().astype(str).map(normalize_code):
        if not code or code in seen:
            continue
        seen.add(code)
        ordered_codes.append(code)
        if limit is not None and len(ordered_codes) >= limit:
            break
    return ordered_codes


def _resolve_strategy_versions(settings: dict, capture_type: str | None = None) -> tuple[list[str], str]:
    versions = available_strategy_versions(settings)
    default_version = normalize_strategy_version(settings.get("default_strategy_version", DEFAULT_STRATEGY_VERSION))
    capture_version = strategy_version_for_capture_type(capture_type)
    if capture_version:
        return [capture_version], capture_version
    if default_version not in versions:
        versions.insert(0, default_version)
    return versions, default_version


def _merge_feature_history(existing_feature_df: pd.DataFrame, fresh_feature_df: pd.DataFrame) -> pd.DataFrame:
    if existing_feature_df.empty:
        return fresh_feature_df.copy()
    if fresh_feature_df.empty:
        return existing_feature_df.copy()

    combined = pd.concat([existing_feature_df.copy(), fresh_feature_df.copy()], ignore_index=True)
    if "signal_date" not in combined.columns or "code" not in combined.columns:
        return fresh_feature_df.copy()

    combined["signal_date"] = pd.to_datetime(combined["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["code"] = combined["code"].astype(str).map(normalize_code)
    if "snapshot_time" in combined.columns:
        combined["snapshot_time"] = combined["snapshot_time"].fillna("").astype(str)
    else:
        combined["snapshot_time"] = ""
    if "rank" in combined.columns:
        combined["rank"] = pd.to_numeric(combined["rank"], errors="coerce")

    combined = combined.dropna(subset=["signal_date"]).copy()
    combined = combined[combined["code"].ne("")].copy()
    combined = combined.sort_values(["signal_date", "snapshot_time", "rank"], na_position="last")
    combined = combined.drop_duplicates(["signal_date", "code"], keep="last")
    return combined.sort_values(["signal_date", "rank"], na_position="last").reset_index(drop=True)


def _latest_popularity_slice(capture_type: str | None = None) -> pd.DataFrame:
    popularity_df = load_popularity()
    if popularity_df.empty or "signal_date" not in popularity_df.columns:
        return pd.DataFrame()

    working = popularity_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working = working.dropna(subset=["signal_date"])
    if working.empty:
        return pd.DataFrame()

    latest_date = working["signal_date"].max()
    latest = working[working["signal_date"].astype(str).eq(str(latest_date))].copy()
    if latest.empty:
        return pd.DataFrame()

    if capture_type:
        matched = latest[latest["capture_type"].fillna("").astype(str).eq(str(capture_type))].copy()
        if not matched.empty:
            latest = matched

    if "snapshot_time" in latest.columns:
        latest["snapshot_time"] = latest["snapshot_time"].fillna("").astype(str)
        latest_snapshot = latest["snapshot_time"][latest["snapshot_time"].ne("")].max()
        if latest_snapshot:
            latest = latest[latest["snapshot_time"].eq(latest_snapshot)].copy()
    return latest.sort_values("rank", na_position="last").reset_index(drop=True)


def run_pipeline(
    native_fetch: bool = True,
    capture_type: str | None = None,
    snapshot_time: str | None = None,
    force_refresh_prices: bool = False,
) -> dict:
    settings = load_settings()
    ensure_layout()
    resolved_capture_type = capture_type or str(settings.get("default_capture_type", "post_close"))
    strategy_versions, default_strategy_version = _resolve_strategy_versions(settings, capture_type=capture_type)

    result: dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "strategy_versions": strategy_versions,
        "default_strategy_version": default_strategy_version,
    }

    if native_fetch:
        fetch_guard = should_skip_market_fetch(resolved_capture_type)
        if fetch_guard["skip"]:
            result["data"] = {
                "status": "skipped_market_closed",
                "source": "local_cache",
                "reason": fetch_guard["reason"],
                "skip_reason_code": fetch_guard.get("skip_reason_code", ""),
                "expected_signal_date": fetch_guard["expected_signal_date"],
                "popularity_rows": 0,
            }
        else:
            result["data"] = run_native_fetch(
                capture_type=resolved_capture_type,
                snapshot_time=snapshot_time,
                top_n=int(settings.get("top_n", 100)),
                refresh_prices=bool(settings.get("refresh_price_cache", True)),
                force_refresh_prices=force_refresh_prices,
            )
    else:
        result["data"] = {"status": "skipped"}

    data_status = str((result.get("data", {}) or {}).get("status", ""))
    skip_reason_code = str((result.get("data", {}) or {}).get("skip_reason_code", ""))

    should_prepare_intraday_feature_cache = (
        native_fetch
        and resolved_capture_type.startswith("intraday_")
        and data_status == "ok"
        and skip_reason_code not in {"holiday", "weekend"}
    )
    if should_prepare_intraday_feature_cache:
        latest_popularity = _latest_popularity_slice(capture_type=resolved_capture_type)
        if not latest_popularity.empty and "code" in latest_popularity.columns:
            trade_date = str(latest_popularity["signal_date"].max())
            snapshot_time = ""
            if "snapshot_time" in latest_popularity.columns:
                snapshot_time = str(latest_popularity["snapshot_time"].fillna("").astype(str).max() or "")
            if trade_date:
                result["intraday_feature_cache"] = warm_intraday_cache(
                    latest_popularity["code"].dropna().astype(str).map(normalize_code).tolist(),
                    trade_date=trade_date,
                    capture_type=resolved_capture_type,
                    snapshot_time=snapshot_time or None,
                    refresh_snapshot=True,
                    refresh_bars=False,
                    force_refresh_snapshot=True,
                    force_refresh_bars=False,
                )

    popularity_df = load_popularity()
    strategy_results: dict[str, dict[str, object]] = {}
    feature_frames: dict[str, pd.DataFrame] = {}
    existing_feature_history = read_csv_safely(FEATURES_CSV)
    shared_feature_history = pd.DataFrame()
    for strategy_version in strategy_versions:
        feature_df = build_daily_features(
            popularity_df=popularity_df,
            strategy_version=strategy_version,
        )
        if strategy_version == default_strategy_version:
            feature_df = _merge_feature_history(existing_feature_history, feature_df)
            shared_feature_history = feature_df.copy()
        elif not shared_feature_history.empty:
            current_date_count = feature_df["signal_date"].nunique() if not feature_df.empty and "signal_date" in feature_df.columns else 0
            shared_date_count = shared_feature_history["signal_date"].nunique() if "signal_date" in shared_feature_history.columns else 0
            if current_date_count < shared_date_count:
                feature_df = shared_feature_history.copy()
        feature_frames[strategy_version] = feature_df
        strategy_results[strategy_version] = {
            "features": {
                "rows": len(feature_df),
                "date_count": feature_df["signal_date"].nunique() if not feature_df.empty else 0,
                "code_count": feature_df["code"].nunique() if not feature_df.empty else 0,
                "capture_types": sorted(feature_df["capture_type"].dropna().astype(str).unique().tolist()) if not feature_df.empty and "capture_type" in feature_df.columns else [],
            }
        }

    default_feature_df = feature_frames.get(default_strategy_version, pd.DataFrame())
    save_daily_features(default_feature_df)
    result["features"] = strategy_results.get(default_strategy_version, {}).get("features", {})

    should_refresh_market_cache = (
        native_fetch
        and data_status in {"ok", "skipped_market_closed", "skipped_existing_snapshot"}
        and skip_reason_code not in {"holiday", "weekend"}
        and bool(settings.get("refresh_market_cache", True))
    )
    if should_refresh_market_cache:
        result["market_cache"] = warm_market_index_cache(force_refresh=force_refresh_prices)

    market_regime_df = build_market_regime(default_feature_df)
    save_market_regime(market_regime_df)
    latest_market = market_regime_df.tail(1).to_dict("records")[0] if not market_regime_df.empty else {}
    result["market_regime"] = {
        "rows": len(market_regime_df),
        "latest_date": latest_market.get("signal_date"),
        "latest_regime": latest_market.get("market_regime"),
        "latest_1d_pct": latest_market.get("market_1d_pct"),
        "latest_5d_pct": latest_market.get("market_5d_pct"),
    }

    min_score = float(settings.get("signal_min_score", 60))
    followup_days = list(settings.get("followup_days", [1, 3, 5, 10]))
    latest_push_limit = _optional_positive_int(settings.get("latest_push_limit"))
    strong_threshold = float(settings.get("strong_return_threshold_pct", 15))

    signal_frames: dict[str, pd.DataFrame] = {}
    for strategy_version in strategy_versions:
        signal_df = build_signals(
            feature_frames[strategy_version],
            min_score=min_score,
            market_regime_df=market_regime_df,
            strategy_version=strategy_version,
        )
        save_signals(signal_df, strategy_version=strategy_version)
        signal_frames[strategy_version] = signal_df
        strategy_results[strategy_version]["signals"] = {
            "rows": len(signal_df),
            "pushed_rows": int(signal_df["is_pushed"].sum()) if not signal_df.empty and "is_pushed" in signal_df.columns else 0,
        }

    default_signal_df = signal_frames.get(default_strategy_version, pd.DataFrame())
    result["signals"] = strategy_results.get(default_strategy_version, {}).get("signals", {})

    should_refresh_followup_prices = (
        native_fetch
        and data_status in {"ok", "skipped_market_closed", "skipped_existing_snapshot"}
        and skip_reason_code not in {"holiday", "weekend"}
        and bool(settings.get("refresh_price_cache", True))
        and any(not frame.empty for frame in signal_frames.values())
    )
    if should_refresh_followup_prices:
        refresh_codes: set[str] = set()
        for strategy_version, signal_df in signal_frames.items():
            refresh_codes.update(
                _followup_refresh_codes(
                    signal_df,
                    followup_days=followup_days,
                    strategy_version=strategy_version,
                )
            )
        result["followup_price_cache"] = warm_stock_price_cache(
            sorted(refresh_codes),
            force_refresh=force_refresh_prices,
        )

    followup_frames: dict[str, pd.DataFrame] = {}
    for strategy_version in strategy_versions:
        signal_df = signal_frames[strategy_version]
        followup_df = build_followups(signal_df, days=followup_days, strategy_version=strategy_version)
        save_followups(followup_df, strategy_version=strategy_version)
        followup_frames[strategy_version] = followup_df
        strategy_results[strategy_version]["followups"] = {
            "rows": len(followup_df),
            "latest_observed_days": int(followup_df["observed_days"].max()) if not followup_df.empty and "observed_days" in followup_df.columns else 0,
        }
        strategy_results[strategy_version]["reports"] = build_reports(
            signal_df=signal_df,
            followup_df=followup_df,
            latest_push_limit=latest_push_limit,
            strong_return_threshold_pct=strong_threshold,
            strategy_version=strategy_version,
        )

    default_followup_df = followup_frames.get(default_strategy_version, pd.DataFrame())
    result["followups"] = strategy_results.get(default_strategy_version, {}).get("followups", {})
    result["reports"] = strategy_results.get(default_strategy_version, {}).get("reports", {})
    result["strategies"] = strategy_results

    should_refresh_intraday_cache = (
        native_fetch
        and resolved_capture_type == "post_close"
        and data_status in {"ok", "skipped_market_closed", "skipped_existing_snapshot"}
        and skip_reason_code not in {"holiday", "weekend", "before_capture"}
        and bool(settings.get("refresh_intraday_cache", True))
        and any(not frame.empty for frame in signal_frames.values())
    )
    if should_refresh_intraday_cache:
        latest_signal_slice = _latest_signal_slice(default_signal_df)
        trade_date = str(latest_signal_slice["signal_date"].max()) if not latest_signal_slice.empty and "signal_date" in latest_signal_slice.columns else ""
        refresh_codes: set[str] = set()
        for signal_df in signal_frames.values():
            refresh_codes.update(
                _intraday_refresh_codes(
                    signal_df,
                    pushed_only=bool(settings.get("intraday_cache_push_only", True)),
                    limit=_optional_positive_int(settings.get("intraday_cache_limit")),
                )
            )
        if trade_date and refresh_codes:
            expected_market_date = latest_expected_market_date().strftime("%Y-%m-%d")
            same_day_post_close = trade_date == expected_market_date
            result["intraday_cache"] = warm_intraday_cache(
                sorted(refresh_codes),
                trade_date=trade_date,
                capture_type=resolved_capture_type,
                refresh_snapshot=same_day_post_close,
                refresh_bars=True,
                force_refresh_snapshot=same_day_post_close,
                force_refresh_bars=force_refresh_prices,
            )

    result["finished_at"] = datetime.now().isoformat(timespec="seconds")
    return result
