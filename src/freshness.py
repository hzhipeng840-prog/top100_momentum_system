from __future__ import annotations

import pandas as pd

from src.trading_calendar import latest_expected_market_date, previous_a_share_trading_day


DEFAULT_SETTLEMENT_FRESHNESS_MIN_RATIO = 0.95


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _normalize_date_text(value: object) -> str:
    parsed = pd.to_datetime(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).normalize().strftime("%Y-%m-%d")


def _normalize_date(value: object) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_localize(None)
    return parsed.normalize()


def _latest_market_row(market_regime_df: pd.DataFrame, expected_market_date: pd.Timestamp) -> pd.Series | None:
    if market_regime_df is None or market_regime_df.empty or "signal_date" not in market_regime_df.columns:
        return None

    working = market_regime_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce")
    working = working.dropna(subset=["signal_date"]).copy()
    if working.empty:
        return None

    expected_date = _normalize_date(expected_market_date)
    working = working[working["signal_date"].dt.normalize() <= expected_date].copy()
    if working.empty:
        working = market_regime_df.copy()
        working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce")
        working = working.dropna(subset=["signal_date"]).copy()
        if working.empty:
            return None

    return working.sort_values("signal_date").iloc[-1]


def build_data_freshness_report(
    followup_df: pd.DataFrame | None,
    market_regime_df: pd.DataFrame | None = None,
    *,
    min_settlement_ratio: float = DEFAULT_SETTLEMENT_FRESHNESS_MIN_RATIO,
    now: object | None = None,
) -> dict[str, object]:
    expected_market_date = _normalize_date(latest_expected_market_date(now))
    settlement_date = _normalize_date(previous_a_share_trading_day(expected_market_date))
    settlement_date_text = settlement_date.strftime("%Y-%m-%d")

    report: dict[str, object] = {
        "status": "pending",
        "is_fresh": False,
        "expected_market_date": expected_market_date.strftime("%Y-%m-%d"),
        "settlement_date": settlement_date_text,
        "settlement_row_count": 0,
        "settled_1d_row_count": 0,
        "settled_1d_ratio": None,
        "market_price_date": None,
        "market_lag_days": None,
        "reason": "",
        "summary": "",
    }

    reasons: list[str] = []
    followups = pd.DataFrame() if followup_df is None else followup_df.copy()
    if not followups.empty and "signal_date" in followups.columns:
        followups["signal_date"] = pd.to_datetime(followups["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        settlement_rows = followups[followups["signal_date"].eq(settlement_date_text)].copy()
    else:
        settlement_rows = pd.DataFrame()

    settlement_row_count = int(len(settlement_rows))
    report["settlement_row_count"] = settlement_row_count

    if settlement_row_count == 0:
        reasons.append(f"未找到 {settlement_date_text} 的结算记录")
        report["status"] = "missing"
    else:
        if "settled_1d" in settlement_rows.columns:
            settled_count = int(_truthy_mask(settlement_rows["settled_1d"]).sum())
        else:
            settled_count = 0
        settled_ratio = settled_count / settlement_row_count if settlement_row_count else 0.0
        report["settled_1d_row_count"] = settled_count
        report["settled_1d_ratio"] = round(float(settled_ratio), 4)
        if settled_ratio < float(min_settlement_ratio):
            reasons.append(
                f"{settlement_date_text} 的 1日收益仅结算 {settled_count}/{settlement_row_count}，低于 {min_settlement_ratio:.0%}"
            )
            report["status"] = "stale"
        else:
            report["status"] = "fresh"

    market_row = _latest_market_row(market_regime_df if market_regime_df is not None else pd.DataFrame(), expected_market_date)
    if market_row is not None:
        market_price_date = _normalize_date_text(market_row.get("market_price_date") or market_row.get("signal_date"))
        market_lag_days = pd.to_numeric(pd.Series([market_row.get("market_lag_days")]), errors="coerce").iloc[0]
        report["market_price_date"] = market_price_date or None
        report["market_lag_days"] = None if pd.isna(market_lag_days) else int(market_lag_days)
        if pd.isna(market_lag_days):
            reasons.append("市场环境缓存缺少滞后信息")
            if report["status"] == "fresh":
                report["status"] = "stale"
        elif int(market_lag_days) > 0:
            reasons.append(f"市场环境仍滞后 {int(market_lag_days)} 天")
            report["status"] = "stale"
    else:
        reasons.append("市场环境缓存缺失")
        if report["status"] == "fresh":
            report["status"] = "stale"

    if report["status"] == "fresh":
        summary = (
            f"数据新鲜度正常：{settlement_date_text} 1日收益已结算。"
            f" 市场缓存已追到 {report['market_price_date'] or '-'}。"
        )
    elif report["status"] == "missing":
        summary = f"数据未结算：暂未找到 {settlement_date_text} 的 1日收益。"
    else:
        summary = "数据新鲜度待处理：" + "；".join(reasons)

    report["reason"] = "；".join(reasons)
    report["summary"] = summary
    report["is_fresh"] = report["status"] == "fresh"
    return report
