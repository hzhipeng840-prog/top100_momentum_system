from __future__ import annotations

import os
import time
from dataclasses import dataclass

import pandas as pd

from src.paths import FEATURES_CSV, MARKET_REGIME_CSV, RAW_INDEX_PRICE_DIR
from src.trading_calendar import latest_expected_market_date
from src.utils import pct_change, read_csv_safely, write_csv


@dataclass(frozen=True)
class IndexSpec:
    symbol: str
    code: str
    name: str
    weight: float


INDEX_SPECS = [
    IndexSpec(symbol="sh000001", code="000001", name="上证指数", weight=0.40),
    IndexSpec(symbol="sz399001", code="399001", name="深证成指", weight=0.35),
    IndexSpec(symbol="sz399006", code="399006", name="创业板指", weight=0.25),
]

MARKET_REGIME_COLUMNS = [
    "signal_date",
    "market_regime",
    "market_score",
    "market_1d_pct",
    "market_5d_pct",
    "market_lag_days",
    "market_price_date",
    "market_source",
    "sh_1d_pct",
    "sz_1d_pct",
    "cyb_1d_pct",
    "sh_5d_pct",
    "sz_5d_pct",
    "cyb_5d_pct",
]


def _disable_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def _cache_path(symbol: str):
    return RAW_INDEX_PRICE_DIR / f"{symbol}.csv"


def _expected_market_date() -> pd.Timestamp:
    return latest_expected_market_date().tz_localize(None).normalize()


def _latest_index_date(index_df: pd.DataFrame) -> pd.Timestamp | None:
    if index_df.empty or "date" not in index_df.columns:
        return None
    latest = pd.to_datetime(index_df["date"], errors="coerce").dropna().max()
    if pd.isna(latest):
        return None
    return pd.Timestamp(latest).normalize()


def _normalize_index_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "日期": "date",
        "时间": "date",
        "date": "date",
        "开盘": "open",
        "open": "open",
        "收盘": "close",
        "close": "close",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "成交量": "volume",
        "volume": "volume",
        "成交额": "amount",
        "amount": "amount",
    }
    result = df.rename(columns=rename_map).copy()
    if "date" not in result.columns or "close" not in result.columns:
        return pd.DataFrame()

    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in ["open", "high", "low"]:
        result[column] = result[column].fillna(result["close"])

    columns = ["date", "open", "close", "high", "low", "volume", "amount"]
    return (
        result[columns]
        .dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .tail(1200)
        .reset_index(drop=True)
    )


def _normalize_index_spot_df(df: pd.DataFrame | None, target_date: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "\u4ee3\u7801": "symbol",
        "\u540d\u79f0": "name",
        "\u6700\u65b0\u4ef7": "close",
        "\u4eca\u5f00": "open",
        "\u6700\u9ad8": "high",
        "\u6700\u4f4e": "low",
        "\u6210\u4ea4\u91cf": "volume",
        "\u6210\u4ea4\u989d": "amount",
        "symbol": "symbol",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
    }
    result = df.rename(columns=rename_map).copy()
    required = {"symbol", "close"}
    if not required.issubset(result.columns):
        return pd.DataFrame()

    result["symbol"] = result["symbol"].astype(str).str.strip()
    result["date"] = pd.Timestamp(target_date).normalize()
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        if column not in result.columns:
            result[column] = None
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in ["open", "high", "low"]:
        result[column] = result[column].fillna(result["close"])

    columns = ["symbol", "date", "open", "close", "high", "low", "volume", "amount"]
    return (
        result[columns]
        .dropna(subset=["symbol", "date", "close"])
        .drop_duplicates("symbol", keep="last")
        .reset_index(drop=True)
    )


def _refresh_start_date(cached_df: pd.DataFrame) -> str:
    if cached_df.empty or "date" not in cached_df.columns:
        return "20230101"
    latest = pd.to_datetime(cached_df["date"], errors="coerce").dropna().max()
    if pd.isna(latest):
        return "20230101"
    start = max(pd.Timestamp("20230101"), pd.Timestamp(latest).normalize() - pd.Timedelta(days=40))
    return start.strftime("%Y%m%d")


def _cache_is_current(cached_df: pd.DataFrame) -> bool:
    return _cache_reaches_date(cached_df, _expected_market_date())


def _cache_reaches_date(cached_df: pd.DataFrame, target_date: pd.Timestamp) -> bool:
    latest = _latest_index_date(cached_df)
    if latest is None:
        return False
    return latest >= pd.Timestamp(target_date).normalize()


def _remote_index_data(ak, spec: IndexSpec, start_date: str) -> pd.DataFrame:
    attempts = [
        lambda: ak.stock_zh_index_daily(symbol=spec.symbol),
        lambda: ak.index_zh_a_hist(symbol=spec.code, period="daily", start_date=start_date, end_date="20300101"),
    ]
    for remote_call in attempts:
        try:
            remote_df = _normalize_index_df(remote_call())
            if remote_df.empty:
                continue
            start_ts = pd.Timestamp(start_date)
            return remote_df[remote_df["date"] >= start_ts].copy()
        except Exception:
            time.sleep(0.5)
    return pd.DataFrame()


def _remote_index_spot_data(ak, target_date: pd.Timestamp) -> pd.DataFrame:
    try:
        return _normalize_index_spot_df(ak.stock_zh_index_spot_sina(), target_date)
    except Exception:
        time.sleep(0.5)
    return pd.DataFrame()


def _merge_spot_row(index_df: pd.DataFrame, spot_df: pd.DataFrame, spec: IndexSpec) -> pd.DataFrame:
    if spot_df.empty or "symbol" not in spot_df.columns:
        return index_df
    matched = spot_df[spot_df["symbol"].astype(str).eq(spec.symbol)].copy()
    if matched.empty:
        return index_df
    matched = matched.drop(columns=["symbol"])
    combined = pd.concat([index_df, matched], ignore_index=True) if not index_df.empty else matched
    return _normalize_index_df(combined)


def _index_frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    normalized_left = _normalize_index_df(left).reset_index(drop=True)
    normalized_right = _normalize_index_df(right).reset_index(drop=True)
    return normalized_left.equals(normalized_right)


def fetch_index_price(symbol: str, force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    spec = next((item for item in INDEX_SPECS if item.symbol == symbol), None)
    if spec is None:
        return pd.DataFrame(), "missing_symbol"

    cache_path = _cache_path(symbol)
    cached_df = _normalize_index_df(read_csv_safely(cache_path))
    freshness_target = _expected_market_date()
    if not force_refresh and not cached_df.empty and _cache_reaches_date(cached_df, freshness_target):
        return cached_df, "cache"

    try:
        import akshare as ak
    except ImportError:
        return cached_df, "missing_akshare" if not cached_df.empty else "missing_akshare_no_cache"

    _disable_proxy_env()
    remote_daily_df = _remote_index_data(ak, spec, _refresh_start_date(cached_df))
    combined = pd.concat([cached_df, remote_daily_df], ignore_index=True) if not remote_daily_df.empty else cached_df.copy()
    combined = _normalize_index_df(combined)

    used_spot = False
    if not _cache_reaches_date(combined, freshness_target):
        spot_df = _remote_index_spot_data(ak, freshness_target)
        supplemented = _merge_spot_row(combined, spot_df, spec)
        used_spot = not _index_frames_equal(combined, supplemented)
        combined = supplemented

    if not combined.empty and not _index_frames_equal(cached_df, combined):
        write_csv(combined, cache_path)

    if _cache_reaches_date(combined, freshness_target):
        return combined, "remote" if (not remote_daily_df.empty or used_spot) else "cache"
    if not combined.empty:
        return combined, "stale_cache"
    return combined, "missing_remote"


def warm_market_index_cache(force_refresh: bool = False) -> dict:
    stats = {
        "requested": len(INDEX_SPECS),
        "cache": 0,
        "remote": 0,
        "stale_cache": 0,
        "missing": 0,
        "target_dir": str(RAW_INDEX_PRICE_DIR),
    }
    for spec in INDEX_SPECS:
        _, source = fetch_index_price(spec.symbol, force_refresh=force_refresh)
        if source in stats:
            stats[source] += 1
        elif source.startswith("missing"):
            stats["missing"] += 1
        else:
            stats["stale_cache"] += 1
    return stats


def _weighted_average(values: list[tuple[float | None, float]]) -> float | None:
    valid = [(float(value), float(weight)) for value, weight in values if value is not None and pd.notna(value)]
    if not valid:
        return None
    total_weight = sum(weight for _, weight in valid)
    if total_weight <= 0:
        return None
    return round(sum(value * weight for value, weight in valid) / total_weight, 2)


def _latest_common_date(index_frames: dict[str, pd.DataFrame], target_date: pd.Timestamp) -> pd.Timestamp | None:
    common_dates: set[pd.Timestamp] | None = None
    for index_df in index_frames.values():
        if index_df.empty or "date" not in index_df.columns:
            return None
        valid_dates = set(pd.to_datetime(index_df["date"], errors="coerce").dropna().dt.normalize())
        valid_dates = {date for date in valid_dates if date <= target_date}
        if not valid_dates:
            return None
        common_dates = valid_dates if common_dates is None else common_dates.intersection(valid_dates)
        if not common_dates:
            return None
    return max(common_dates) if common_dates else None


def _index_move_for_date(index_df: pd.DataFrame, base_date: pd.Timestamp) -> dict:
    if index_df.empty:
        return {"one_day": None, "five_day": None, "price_date": None}

    base_date = pd.Timestamp(base_date).normalize()
    history_df = index_df[index_df["date"] <= base_date].copy()
    if history_df.empty:
        return {"one_day": None, "five_day": None, "price_date": None}

    current = history_df.iloc[-1]
    current_close = pd.to_numeric(pd.Series([current.get("close")]), errors="coerce").iloc[0]
    previous_close = None
    five_day_close = None
    if len(history_df) >= 2:
        previous_close = pd.to_numeric(pd.Series([history_df.iloc[-2].get("close")]), errors="coerce").iloc[0]
    if len(history_df) >= 6:
        five_day_close = pd.to_numeric(pd.Series([history_df.iloc[-6].get("close")]), errors="coerce").iloc[0]

    price_date = pd.Timestamp(current.get("date")).normalize()
    return {
        "one_day": pct_change(float(current_close), float(previous_close)) if pd.notna(current_close) and pd.notna(previous_close) else None,
        "five_day": pct_change(float(current_close), float(five_day_close)) if pd.notna(current_close) and pd.notna(five_day_close) else None,
        "price_date": price_date,
    }


def _classify_regime(one_day: float | None, five_day: float | None, lag_days: int | None, index_moves: list[dict]) -> str:
    if one_day is None and five_day is None:
        return "未知"
    if lag_days is not None and lag_days > 5:
        return "未知"

    weak_count = sum(1 for item in index_moves if item.get("one_day") is not None and float(item["one_day"]) <= -1.2)
    strong_count = sum(1 for item in index_moves if item.get("one_day") is not None and float(item["one_day"]) >= 1.0)
    one_day_value = float(one_day) if one_day is not None else 0.0
    five_day_value = float(five_day) if five_day is not None else 0.0

    if one_day_value <= -1.2 or five_day_value <= -3.0 or weak_count >= 2:
        return "弱势"
    if one_day_value >= 1.0 or five_day_value >= 2.5 or strong_count >= 2:
        return "强势"
    return "震荡"


def _market_score(one_day: float | None, five_day: float | None) -> float | None:
    if one_day is None and five_day is None:
        return None
    value = 50.0
    if one_day is not None:
        value += float(one_day) * 8.0
    if five_day is not None:
        value += float(five_day) * 3.0
    return round(min(max(value, 0.0), 100.0), 2)


def build_market_regime(feature_df: pd.DataFrame | None = None) -> pd.DataFrame:
    features = read_csv_safely(FEATURES_CSV) if feature_df is None else feature_df.copy()
    if features.empty or "signal_date" not in features.columns:
        return pd.DataFrame(columns=MARKET_REGIME_COLUMNS)

    dates = sorted(pd.to_datetime(features["signal_date"], errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    if not dates:
        return pd.DataFrame(columns=MARKET_REGIME_COLUMNS)

    index_frames = {spec.symbol: _normalize_index_df(read_csv_safely(_cache_path(spec.symbol))) for spec in INDEX_SPECS}
    rows: list[dict] = []
    for signal_date in dates:
        target_date = pd.Timestamp(signal_date).normalize()
        common_date = _latest_common_date(index_frames, target_date)
        index_moves = []
        for spec in INDEX_SPECS:
            move = _index_move_for_date(index_frames[spec.symbol], common_date) if common_date is not None else {"one_day": None, "five_day": None, "price_date": None}
            move["spec"] = spec
            index_moves.append(move)

        market_1d = _weighted_average([(item["one_day"], item["spec"].weight) for item in index_moves])
        market_5d = _weighted_average([(item["five_day"], item["spec"].weight) for item in index_moves])
        price_dates = [item["price_date"] for item in index_moves if item["price_date"] is not None]
        latest_price_date = max(price_dates) if price_dates else None
        lag_days = None
        if latest_price_date is not None:
            lag_days = int((target_date - latest_price_date).days)

        rows.append(
            {
                "signal_date": signal_date,
                "market_regime": _classify_regime(market_1d, market_5d, lag_days, index_moves),
                "market_score": _market_score(market_1d, market_5d),
                "market_1d_pct": market_1d,
                "market_5d_pct": market_5d,
                "market_lag_days": lag_days,
                "market_price_date": latest_price_date.strftime("%Y-%m-%d") if latest_price_date is not None else None,
                "market_source": "common_index_cache" if price_dates else "missing_index_cache",
                "sh_1d_pct": index_moves[0]["one_day"],
                "sz_1d_pct": index_moves[1]["one_day"],
                "cyb_1d_pct": index_moves[2]["one_day"],
                "sh_5d_pct": index_moves[0]["five_day"],
                "sz_5d_pct": index_moves[1]["five_day"],
                "cyb_5d_pct": index_moves[2]["five_day"],
            }
        )

    result = pd.DataFrame(rows)
    for column in MARKET_REGIME_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result[MARKET_REGIME_COLUMNS].reset_index(drop=True)


def attach_market_regime(df: pd.DataFrame, market_regime_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df.empty or "signal_date" not in df.columns:
        return df

    result = df.copy()
    regime_df = build_market_regime(result) if market_regime_df is None else market_regime_df.copy()
    if regime_df.empty:
        result["market_regime"] = "未知"
        result["market_1d_pct"] = None
        result["market_5d_pct"] = None
        result["relative_1d_pct"] = None
        result["relative_5d_pct"] = None
        return result

    result["signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    regime_df["signal_date"] = pd.to_datetime(regime_df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result = result.merge(regime_df, on="signal_date", how="left")
    result["market_regime"] = result["market_regime"].fillna("未知")

    result["day_return_pct"] = pd.to_numeric(result.get("day_return_pct"), errors="coerce")
    result["pre5_return_pct"] = pd.to_numeric(result.get("pre5_return_pct"), errors="coerce")
    result["market_1d_pct"] = pd.to_numeric(result.get("market_1d_pct"), errors="coerce")
    result["market_5d_pct"] = pd.to_numeric(result.get("market_5d_pct"), errors="coerce")
    result["relative_1d_pct"] = (result["day_return_pct"] - result["market_1d_pct"]).round(2)
    result["relative_5d_pct"] = (result["pre5_return_pct"] - result["market_5d_pct"]).round(2)
    return result


def save_market_regime(market_regime_df: pd.DataFrame) -> None:
    write_csv(market_regime_df, MARKET_REGIME_CSV)
