from __future__ import annotations

import pandas as pd

from src.market_data import load_price_data
from src.paths import followups_csv_for, signals_csv_for
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, normalize_strategy_version
from src.utils import normalize_code, parse_number, pct_change, read_csv_safely, write_csv


BASE_COLUMNS = [
    "strategy_version",
    "signal_date",
    "code",
    "name",
    "rank",
    "push_level",
    "emotion_score",
    "is_pushed",
    "capture_type",
    "snapshot_time",
    "reasons",
    "risks",
    "price_date",
    "signal_close",
    "latest_price_date",
    "observed_days",
    "latest_return_pct",
    "tail_next_open_pct",
    "tail_next_close_pct",
    "tail_next_max_gain_pct",
    "tail_next_max_drawdown_pct",
    "settled_tail_next_day",
]


def _empty_result(days: list[int]) -> dict:
    result: dict[str, object] = {}
    for day in days:
        result[f"return_{day}d_pct"] = None
        result[f"max_gain_{day}d_pct"] = None
        result[f"max_drawdown_{day}d_pct"] = None
        result[f"settled_{day}d"] = False
    return result


def calculate_followup(row: pd.Series, days: list[int], price_cache: dict[str, pd.DataFrame] | None = None) -> dict:
    code = normalize_code(row.get("code"))
    if price_cache is not None and code in price_cache:
        price_df = price_cache[code]
    else:
        price_df = load_price_data(code)
        if price_cache is not None:
            price_cache[code] = price_df
    signal_close = parse_number(row.get("close"))
    price_date = pd.to_datetime(row.get("price_date"), errors="coerce")

    base = {
        "signal_close": signal_close,
        "latest_price_date": None,
        "observed_days": 0,
        "latest_return_pct": None,
        "tail_next_open_pct": None,
        "tail_next_close_pct": None,
        "tail_next_max_gain_pct": None,
        "tail_next_max_drawdown_pct": None,
        "settled_tail_next_day": False,
    }
    base.update(_empty_result(days))

    if price_df.empty or signal_close is None or signal_close == 0 or pd.isna(price_date):
        return base

    future_df = price_df[price_df["date"] > pd.Timestamp(price_date).normalize()].copy().reset_index(drop=True)
    if future_df.empty:
        return base

    latest = future_df.iloc[-1]
    latest_close = parse_number(latest.get("close"))
    base["latest_price_date"] = pd.Timestamp(latest.get("date")).strftime("%Y-%m-%d")
    base["observed_days"] = len(future_df)
    base["latest_return_pct"] = pct_change(latest_close, signal_close)

    next_day = future_df.iloc[0]
    next_open = parse_number(next_day.get("open"))
    next_close = parse_number(next_day.get("close"))
    next_high = parse_number(next_day.get("high"))
    next_low = parse_number(next_day.get("low"))
    base["tail_next_open_pct"] = pct_change(next_open, signal_close)
    base["tail_next_close_pct"] = pct_change(next_close, signal_close)
    base["tail_next_max_gain_pct"] = pct_change(next_high, signal_close)
    base["tail_next_max_drawdown_pct"] = pct_change(next_low, signal_close)
    base["settled_tail_next_day"] = True

    for day in days:
        if len(future_df) < day:
            continue
        window_df = future_df.iloc[:day]
        end_close = parse_number(window_df.iloc[-1].get("close"))
        max_high = pd.to_numeric(window_df["high"], errors="coerce").dropna().max()
        min_low = pd.to_numeric(window_df["low"], errors="coerce").dropna().min()
        base[f"return_{day}d_pct"] = pct_change(end_close, signal_close)
        base[f"max_gain_{day}d_pct"] = pct_change(float(max_high), signal_close) if pd.notna(max_high) else None
        base[f"max_drawdown_{day}d_pct"] = pct_change(float(min_low), signal_close) if pd.notna(min_low) else None
        base[f"settled_{day}d"] = True
    return base


def build_followups(
    signal_df: pd.DataFrame | None = None,
    days: list[int] | None = None,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    strategy_version = normalize_strategy_version(strategy_version)
    days = days or [1, 3, 5, 10]
    df = read_csv_safely(signals_csv_for(strategy_version)) if signal_df is None else signal_df.copy()
    if df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    price_cache: dict[str, pd.DataFrame] = {}
    for _, row in df.iterrows():
        followup = calculate_followup(row, days=days, price_cache=price_cache)
        record = {column: row.get(column) for column in BASE_COLUMNS if column not in followup}
        record.setdefault("strategy_version", strategy_version)
        record.update(followup)
        rows.append(record)

    result = pd.DataFrame(rows)
    result["strategy_version"] = strategy_version
    sort_columns = ["signal_date", "emotion_score", "rank"]
    return result.sort_values(sort_columns, ascending=[True, False, True], na_position="last").reset_index(drop=True)


def save_followups(followup_df: pd.DataFrame, strategy_version: str = DEFAULT_STRATEGY_VERSION) -> None:
    write_csv(followup_df, followups_csv_for(strategy_version))
