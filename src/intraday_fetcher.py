from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import os

import pandas as pd

from src.paths import INTRADAY_BAR_DIR, INTRADAY_SNAPSHOT_CSV
from src.trading_calendar import current_market_time
from src.utils import normalize_code, read_csv_safely, write_csv


SNAPSHOT_COLUMNS = [
    "snapshot_time",
    "capture_type",
    "code",
    "name",
    "last_price",
    "open",
    "prev_close",
    "current_return_pct",
    "day_high_so_far",
    "day_low_so_far",
    "volume_so_far",
    "amount_so_far",
    "turnover_pct",
    "volume_ratio",
    "source",
]

BAR_COLUMNS = [
    "datetime",
    "code",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "amount",
    "source",
]

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

INTRADAY_REMOTE_SOURCES = {
    "single_remote",
    "akshare.stock_zh_a_hist_min_em",
    "akshare.stock_zh_a_minute",
}


def _now_text() -> str:
    return current_market_time().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def _without_proxy_env():
    saved = {key: os.environ.get(key) for key in PROXY_ENV_KEYS}
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _first_value(row: pd.Series, names: list[str]) -> object:
    for name in names:
        if name in row.index:
            return row.get(name)
    return None


def _to_number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _normalize_snapshot_row(raw_row: pd.Series, code: str, capture_type: str, snapshot_time: str) -> pd.DataFrame:
    record = {
        "snapshot_time": snapshot_time,
        "capture_type": capture_type,
        "code": normalize_code(code),
        "name": _first_value(raw_row, ["名称", "name"]),
        "last_price": _to_number(_first_value(raw_row, ["最新价", "last_price"])),
        "open": _to_number(_first_value(raw_row, ["今开", "open"])),
        "prev_close": _to_number(_first_value(raw_row, ["昨收", "prev_close"])),
        "current_return_pct": _to_number(_first_value(raw_row, ["涨跌幅", "current_return_pct"])),
        "day_high_so_far": _to_number(_first_value(raw_row, ["最高", "day_high_so_far"])),
        "day_low_so_far": _to_number(_first_value(raw_row, ["最低", "day_low_so_far"])),
        "volume_so_far": _to_number(_first_value(raw_row, ["成交量", "volume_so_far"])),
        "amount_so_far": _to_number(_first_value(raw_row, ["成交额", "amount_so_far"])),
        "turnover_pct": _to_number(_first_value(raw_row, ["换手率", "turnover_pct"])),
        "volume_ratio": _to_number(_first_value(raw_row, ["量比", "volume_ratio"])),
        "source": "akshare.stock_zh_a_spot_em",
    }
    return pd.DataFrame([record], columns=SNAPSHOT_COLUMNS)


def _snapshot_df_from_record(record: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame([{column: record.get(column) for column in SNAPSHOT_COLUMNS}], columns=SNAPSHOT_COLUMNS)


def _fetch_single_stock_snapshot_em(code: str, capture_type: str, snapshot_time: str) -> pd.DataFrame:
    import requests

    market_code = 1 if code.startswith("6") else 0
    with _without_proxy_env():
        response = requests.get(
            "https://push2.eastmoney.com/api/qt/stock/get",
            params={
                "fltt": "2",
                "invt": "2",
                "fields": "f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f168,f170",
                "secid": f"{market_code}.{code}",
            },
            timeout=5,
        )
    response.raise_for_status()
    data = response.json().get("data") or {}
    if not data:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    return _snapshot_df_from_record(
        {
            "snapshot_time": snapshot_time,
            "capture_type": capture_type,
            "code": normalize_code(data.get("f57") or code),
            "name": data.get("f58"),
            "last_price": _to_number(data.get("f43")),
            "open": _to_number(data.get("f46")),
            "prev_close": _to_number(data.get("f60")),
            "current_return_pct": _to_number(data.get("f170")),
            "day_high_so_far": _to_number(data.get("f44")),
            "day_low_so_far": _to_number(data.get("f45")),
            "volume_so_far": _to_number(data.get("f47")),
            "amount_so_far": _to_number(data.get("f48")),
            "turnover_pct": _to_number(data.get("f168")),
            "volume_ratio": _to_number(data.get("f50")),
            "source": "eastmoney.single_stock_quote",
        }
    )


def _snapshot_cache_for_code(code: str) -> pd.DataFrame:
    df = read_csv_safely(INTRADAY_SNAPSHOT_CSV)
    if df.empty or "code" not in df.columns:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    return df[df["code"].astype(str).str.zfill(6).eq(normalize_code(code))].copy()


def fetch_intraday_snapshot(
    code: str,
    capture_type: str = "manual",
    snapshot_time: str | None = None,
    force_refresh: bool = True,
) -> tuple[pd.DataFrame, str]:
    normalized_code = normalize_code(code)
    snapshot_time = snapshot_time or _now_text()
    if not normalized_code:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS), "missing_code"

    if not force_refresh:
        cached = _snapshot_cache_for_code(normalized_code)
        if not cached.empty:
            return cached.tail(1), "cache"

    try:
        snapshot_df = _fetch_single_stock_snapshot_em(normalized_code, capture_type, snapshot_time)
    except Exception as exc:
        cached = _snapshot_cache_for_code(normalized_code)
        return cached.tail(1), f"single_remote_failed:{exc}" if cached.empty else "cache_single_remote_failed"

    if snapshot_df.empty:
        cached = _snapshot_cache_for_code(normalized_code)
        return cached.tail(1), "empty_single_remote" if cached.empty else "cache_empty_single_remote"

    old_df = read_csv_safely(INTRADAY_SNAPSHOT_CSV)
    combined = pd.concat([old_df, snapshot_df], ignore_index=True) if not old_df.empty else snapshot_df
    for column in SNAPSHOT_COLUMNS:
        if column not in combined.columns:
            combined[column] = None
    combined["code"] = combined["code"].astype(str).str.zfill(6)
    combined = combined.drop_duplicates(["snapshot_time", "code", "capture_type"], keep="last")
    combined = combined.sort_values(["snapshot_time", "code"]).reset_index(drop=True)
    write_csv(combined[SNAPSHOT_COLUMNS], INTRADAY_SNAPSHOT_CSV)
    return snapshot_df, "single_remote"


def _bar_cache_path(code: str, trade_date: str, period: str) -> object:
    return INTRADAY_BAR_DIR / f"{normalize_code(code)}_{trade_date}_{period}m.csv"


def _sina_symbol(code: str) -> str:
    normalized_code = normalize_code(code)
    return ("sh" if normalized_code.startswith("6") else "sz") + normalized_code


def _normalize_bar_df(df: pd.DataFrame | None, code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    rename_map = {
        "时间": "datetime",
        "日期": "datetime",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "day": "datetime",
    }
    result = df.rename(columns=rename_map).copy()
    for column in BAR_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce")
    result["code"] = normalize_code(code)
    for column in ["open", "close", "high", "low", "volume", "amount"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["source"] = result["source"].fillna("akshare.stock_zh_a_hist_min_em")
    result = result.dropna(subset=["datetime", "close"]).sort_values("datetime").drop_duplicates("datetime", keep="last")
    result["datetime"] = result["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return result[BAR_COLUMNS].reset_index(drop=True)


def fetch_intraday_bars(
    code: str,
    trade_date: str | None = None,
    period: str = "1",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str]:
    normalized_code = normalize_code(code)
    trade_date = trade_date or current_market_time().strftime("%Y-%m-%d")
    if not normalized_code:
        return pd.DataFrame(columns=BAR_COLUMNS), "missing_code"

    cache_path = _bar_cache_path(normalized_code, trade_date, period)
    cached_df = _normalize_bar_df(read_csv_safely(cache_path), normalized_code)
    if not force_refresh and not cached_df.empty:
        return cached_df, "cache"

    try:
        import akshare as ak
    except ImportError:
        return cached_df, "missing_akshare" if not cached_df.empty else "missing_akshare_no_cache"

    start_date = f"{trade_date} 09:30:00"
    end_date = f"{trade_date} 15:00:00"
    remote_errors: list[str] = []
    try:
        with _without_proxy_env():
            remote_df = ak.stock_zh_a_hist_min_em(
                symbol=normalized_code,
                start_date=start_date,
                end_date=end_date,
                period=period,
                adjust="",
            )
    except Exception as exc:
        remote_df = pd.DataFrame()
        remote_errors.append(f"em_failed:{exc}")

    bar_df = _normalize_bar_df(remote_df, normalized_code)
    if bar_df.empty and period == "1":
        try:
            with _without_proxy_env():
                sina_df = ak.stock_zh_a_minute(symbol=_sina_symbol(normalized_code), period=period, adjust="")
            if not sina_df.empty:
                sina_df = sina_df.copy()
                sina_df["source"] = "akshare.stock_zh_a_minute"
                sina_df = sina_df[pd.to_datetime(sina_df["day"], errors="coerce").dt.strftime("%Y-%m-%d").eq(trade_date)]
            bar_df = _normalize_bar_df(sina_df, normalized_code)
        except Exception as exc:
            remote_errors.append(f"sina_failed:{exc}")

    if bar_df.empty:
        source = "remote_failed:" + " | ".join(remote_errors) if remote_errors else "empty_remote"
        return cached_df, source if cached_df.empty else "cache_" + source

    write_csv(bar_df, cache_path)
    source = str(bar_df["source"].dropna().iloc[0]) if "source" in bar_df.columns and not bar_df["source"].dropna().empty else "remote"
    return bar_df, source


def _cache_result_bucket(source: str) -> str:
    normalized = str(source or "").strip().lower()
    if not normalized:
        return "missing"
    if normalized == "cache" or normalized.startswith("cache_"):
        return "cache"
    if source in INTRADAY_REMOTE_SOURCES:
        return "remote"
    return "missing"


def warm_intraday_cache(
    codes: list[str],
    trade_date: str,
    capture_type: str = "post_close",
    snapshot_time: str | None = None,
    period: str = "1",
    refresh_snapshot: bool = True,
    refresh_bars: bool = True,
    force_refresh_snapshot: bool = False,
    force_refresh_bars: bool = False,
) -> dict[str, object]:
    normalized_codes = sorted(set(normalize_code(code) for code in codes if normalize_code(code)))
    stats: dict[str, object] = {
        "trade_date": trade_date,
        "capture_type": capture_type,
        "requested": len(normalized_codes),
        "snapshot": {
            "requested": len(normalized_codes) if refresh_snapshot else 0,
            "cache": 0,
            "remote": 0,
            "missing": 0,
        },
        "bars": {
            "requested": len(normalized_codes) if refresh_bars else 0,
            "cache": 0,
            "remote": 0,
            "missing": 0,
            "period": period,
        },
    }
    if not normalized_codes:
        return stats

    resolved_snapshot_time = snapshot_time or f"{trade_date} 15:00:00"
    for code in normalized_codes:
        if refresh_snapshot:
            _, snapshot_source = fetch_intraday_snapshot(
                code,
                capture_type=capture_type,
                snapshot_time=resolved_snapshot_time,
                force_refresh=force_refresh_snapshot,
            )
            snapshot_bucket = _cache_result_bucket(snapshot_source)
            stats["snapshot"][snapshot_bucket] += 1

        if refresh_bars:
            _, bar_source = fetch_intraday_bars(
                code,
                trade_date=trade_date,
                period=period,
                force_refresh=force_refresh_bars,
            )
            bar_bucket = _cache_result_bucket(bar_source)
            stats["bars"][bar_bucket] += 1

    return stats
