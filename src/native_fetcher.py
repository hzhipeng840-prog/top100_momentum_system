from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
import requests

from src.paths import RAW_POPULARITY_CSV, RAW_STOCK_PRICE_DIR
from src.trading_calendar import current_market_time, default_signal_date, latest_expected_market_date
from src.utils import normalize_code, read_csv_safely, write_csv


DEFAULT_REMOTE_START_DATE = "20230101"
REFRESH_LOOKBACK_DAYS = 40
STOCK_CACHE_MAX_WORKERS = 6
SEQUENTIAL_REFRESH_THRESHOLD = 8


def _disable_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def _build_direct_session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    return session


def _snapshot_time(signal_date: str, snapshot_time: str | None, capture_type: str) -> str:
    if snapshot_time:
        return snapshot_time
    if capture_type == "post_close":
        return f"{signal_date} 15:00:00"
    return current_market_time().strftime("%Y-%m-%d %H:%M:%S")


def fetch_popularity_top100(
    signal_date: str | None = None,
    capture_type: str = "post_close",
    snapshot_time: str | None = None,
    top_n: int = 100,
) -> pd.DataFrame:
    signal_date = signal_date or default_signal_date(capture_type)
    snapshot_time = _snapshot_time(signal_date, snapshot_time, capture_type)
    url = "https://dq.10jqka.com.cn/fuyao/hot_list_data/out/hot_list/v1/stock"
    params = {
        "stock_type": "a",
        "type": "day",
        "list_type": "value",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/137.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.10jqka.com.cn/",
        "Accept": "application/json, text/plain, */*",
    }

    response = _build_direct_session().get(url, params=params, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()
    rows = []
    for rank, item in enumerate(data.get("data", {}).get("stock_list", [])[:top_n], start=1):
        rows.append(
            {
                "signal_date": signal_date,
                "rank": rank,
                "code": normalize_code(item.get("code")),
                "name": str(item.get("name", "")).strip(),
                "popularity_score": item.get("rate"),
                "source": "10jqka",
                "capture_type": capture_type,
                "snapshot_time": snapshot_time,
            }
        )
    return pd.DataFrame(rows)


def existing_popularity_snapshot(
    signal_date: str | None = None,
    capture_type: str = "post_close",
    snapshot_time: str | None = None,
    min_rows: int = 100,
) -> pd.DataFrame:
    resolved_signal_date = signal_date or default_signal_date(capture_type)
    resolved_snapshot_time = _snapshot_time(resolved_signal_date, snapshot_time, capture_type)
    df = read_csv_safely(RAW_POPULARITY_CSV)
    if df.empty:
        return pd.DataFrame()
    for column in ["signal_date", "capture_type", "snapshot_time", "code"]:
        if column not in df.columns:
            return pd.DataFrame()
    matched = df[
        df["signal_date"].astype(str).eq(str(resolved_signal_date))
        & df["capture_type"].astype(str).eq(str(capture_type))
        & df["snapshot_time"].astype(str).eq(str(resolved_snapshot_time))
    ].copy()
    matched["code"] = matched["code"].apply(normalize_code)
    matched = matched[matched["code"].ne("")]
    if matched["code"].nunique() < min_rows:
        return pd.DataFrame()
    return matched.sort_values("rank", na_position="last").reset_index(drop=True)


def save_popularity_snapshot(snapshot_df: pd.DataFrame, path=RAW_POPULARITY_CSV) -> pd.DataFrame:
    columns = ["signal_date", "rank", "code", "name", "popularity_score", "source", "capture_type", "snapshot_time"]
    if snapshot_df.empty:
        return read_csv_safely(path)

    snapshot_df = snapshot_df.copy()
    for column in columns:
        if column not in snapshot_df.columns:
            snapshot_df[column] = None
    snapshot_df["code"] = snapshot_df["code"].apply(normalize_code)
    snapshot_df["rank"] = pd.to_numeric(snapshot_df["rank"], errors="coerce")
    snapshot_df["popularity_score"] = pd.to_numeric(snapshot_df["popularity_score"], errors="coerce")
    snapshot_df = snapshot_df[columns].dropna(subset=["signal_date", "rank"])
    snapshot_df = snapshot_df[snapshot_df["code"].ne("")]

    old_df = read_csv_safely(path)
    if not old_df.empty:
        for column in columns:
            if column not in old_df.columns:
                old_df[column] = None
        old_df = old_df[columns].copy()
        old_df["code"] = old_df["code"].apply(normalize_code)
        old_df["capture_type"] = old_df["capture_type"].fillna("").astype(str)
        old_df["snapshot_time"] = old_df["snapshot_time"].fillna("").astype(str)
        snapshot_keys = snapshot_df[["signal_date", "capture_type", "snapshot_time"]].drop_duplicates()
        old_df = old_df.merge(snapshot_keys.assign(_replace_snapshot=True), how="left", on=["signal_date", "capture_type", "snapshot_time"])
        old_df = old_df[old_df["_replace_snapshot"].isna()].drop(columns=["_replace_snapshot"])
    combined = pd.concat([old_df, snapshot_df], ignore_index=True) if not old_df.empty else snapshot_df
    combined["code"] = combined["code"].apply(normalize_code)
    combined = combined.drop_duplicates(["signal_date", "code", "capture_type", "snapshot_time"], keep="last")
    combined = combined.sort_values(["signal_date", "snapshot_time", "rank"], na_position="last").reset_index(drop=True)
    write_csv(combined[columns], path)
    return combined[columns]


def _is_beijing_market_code(code: str) -> bool:
    return normalize_code(code).startswith(("4", "8", "92"))


def _add_market_prefix(code: str) -> str:
    code = normalize_code(code)
    if _is_beijing_market_code(code):
        return f"bj{code}"
    if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"sh{code}"
    return f"sz{code}"


def _normalize_price_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "amount": "volume",
    }
    result = df.rename(columns=rename_map).copy()
    required = ["date", "open", "close", "high", "low", "volume"]
    if not set(required).issubset(result.columns):
        return pd.DataFrame()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    for column in ["open", "close", "high", "low", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return (
        result.dropna(subset=required)
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .tail(500)
        .reset_index(drop=True)
    )


def _refresh_start_date(cached_df: pd.DataFrame) -> str:
    if cached_df.empty:
        return DEFAULT_REMOTE_START_DATE
    latest = pd.to_datetime(cached_df["date"], errors="coerce").dropna().max()
    if pd.isna(latest):
        return DEFAULT_REMOTE_START_DATE
    start = max(pd.Timestamp(DEFAULT_REMOTE_START_DATE), pd.Timestamp(latest).normalize() - pd.Timedelta(days=REFRESH_LOOKBACK_DAYS))
    return start.strftime("%Y%m%d")


def _cache_is_current(cached_df: pd.DataFrame) -> bool:
    if cached_df.empty or "date" not in cached_df.columns:
        return False
    latest = pd.to_datetime(cached_df["date"], errors="coerce").dropna().max()
    if pd.isna(latest):
        return False
    return pd.Timestamp(latest).normalize() >= latest_expected_market_date().tz_localize(None)


def fetch_stock_price(code: str, force_refresh: bool = False) -> tuple[pd.DataFrame, str]:
    normalized = normalize_code(code)
    if not normalized:
        return pd.DataFrame(), "missing_code"

    cache_path = RAW_STOCK_PRICE_DIR / f"{normalized}.csv"
    cached_df = _normalize_price_df(read_csv_safely(cache_path))
    if not force_refresh and not cached_df.empty and _cache_is_current(cached_df):
        return cached_df, "cache"

    try:
        import akshare as ak
    except ImportError:
        return cached_df, "missing_akshare" if not cached_df.empty else "missing_akshare_no_cache"

    _disable_proxy_env()
    start_date = _refresh_start_date(cached_df)
    errors: list[str] = []
    remote_df = pd.DataFrame()
    for attempt in range(3):
        try:
            remote_df = _normalize_price_df(
                ak.stock_zh_a_hist_tx(
                    symbol=_add_market_prefix(normalized),
                    start_date=start_date,
                    end_date="20300101",
                    adjust="qfq",
                )
            )
            if not remote_df.empty:
                break
        except Exception as exc:
            errors.append(str(exc))
            time.sleep(1)

    if remote_df.empty:
        for attempt in range(2):
            try:
                remote_df = _normalize_price_df(
                    ak.stock_zh_a_hist(
                        symbol=normalized,
                        period="daily",
                        start_date=start_date,
                        end_date="20300101",
                        adjust="qfq",
                    )
                )
                if not remote_df.empty:
                    break
            except Exception as exc:
                errors.append(str(exc))
                time.sleep(1)

    if remote_df.empty:
        return cached_df, "stale_cache" if not cached_df.empty else f"missing:{errors[-1] if errors else 'unknown'}"

    combined = pd.concat([cached_df, remote_df], ignore_index=True) if not cached_df.empty else remote_df
    combined = _normalize_price_df(combined)
    write_csv(combined, cache_path)
    return combined, "remote"


def _update_price_cache_stats(stats: dict[str, object], source: str) -> None:
    if source in stats:
        stats[source] += 1
    elif source.startswith("missing"):
        stats["missing"] += 1
    else:
        stats["stale_cache"] += 1


def warm_stock_price_cache(codes: list[str], force_refresh: bool = False) -> dict:
    normalized_codes = sorted(set(normalize_code(code) for code in codes if normalize_code(code)))
    stats = {
        "requested": len(normalized_codes),
        "cache": 0,
        "remote": 0,
        "stale_cache": 0,
        "missing": 0,
        "target_dir": str(RAW_STOCK_PRICE_DIR),
    }
    if not normalized_codes:
        return stats

    if len(normalized_codes) <= SEQUENTIAL_REFRESH_THRESHOLD:
        for code in normalized_codes:
            _, source = fetch_stock_price(code, force_refresh=force_refresh)
            _update_price_cache_stats(stats, source)
        return stats

    max_workers = min(STOCK_CACHE_MAX_WORKERS, len(normalized_codes))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, source in executor.map(lambda code: fetch_stock_price(code, force_refresh=force_refresh), normalized_codes):
            _update_price_cache_stats(stats, source)
    return stats


def run_native_fetch(
    signal_date: str | None = None,
    capture_type: str = "post_close",
    snapshot_time: str | None = None,
    top_n: int = 100,
    refresh_prices: bool = True,
    force_refresh_prices: bool = False,
    skip_existing: bool = True,
) -> dict:
    existing_snapshot = existing_popularity_snapshot(
        signal_date=signal_date,
        capture_type=capture_type,
        snapshot_time=snapshot_time,
        min_rows=top_n,
    )
    if skip_existing and not existing_snapshot.empty:
        price_stats = {"requested": 0, "cache": 0, "remote": 0, "stale_cache": 0, "missing": 0}
        if refresh_prices:
            price_stats = warm_stock_price_cache(existing_snapshot["code"].tolist(), force_refresh=force_refresh_prices)
        saved_df = read_csv_safely(RAW_POPULARITY_CSV)
        return {
            "status": "skipped_existing_snapshot",
            "source": "local_cache",
            "reason": "同一采集快照已存在，本次跳过新榜抓取，只刷新缓存和报表。",
            "popularity_rows": len(existing_snapshot),
            "stored_rows": len(saved_df),
            "date_count": saved_df["signal_date"].nunique() if not saved_df.empty and "signal_date" in saved_df.columns else 0,
            "code_count": saved_df["code"].nunique() if not saved_df.empty and "code" in saved_df.columns else 0,
            "popularity_path": str(RAW_POPULARITY_CSV),
            "price_cache": price_stats,
        }

    popularity_df = fetch_popularity_top100(
        signal_date=signal_date,
        capture_type=capture_type,
        snapshot_time=snapshot_time,
        top_n=top_n,
    )
    saved_df = save_popularity_snapshot(popularity_df)
    price_stats = {"requested": 0, "cache": 0, "remote": 0, "stale_cache": 0, "missing": 0}
    if refresh_prices and not popularity_df.empty:
        price_stats = warm_stock_price_cache(popularity_df["code"].tolist(), force_refresh=force_refresh_prices)
    return {
        "status": "ok",
        "source": "native",
        "popularity_rows": len(popularity_df),
        "stored_rows": len(saved_df),
        "date_count": saved_df["signal_date"].nunique() if not saved_df.empty else 0,
        "code_count": saved_df["code"].nunique() if not saved_df.empty else 0,
        "popularity_path": str(RAW_POPULARITY_CSV),
        "price_cache": price_stats,
    }
