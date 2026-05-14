from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from src.paths import V4_CYQ_DIR, V4_FUND_FLOW_DIR, V4_LHB_DIR, V4_NOTICE_CACHE_CSV
from src.utils import normalize_code, parse_number, read_csv_safely, write_csv


NOTICE_CACHE_COLUMNS = ["signal_date", "code", "announcement_summary", "event_summary"]
FUND_FLOW_CACHE_COLUMNS = ["signal_date", "code", "capital_flow_pct", "capital_flow_signal"]
LHB_CACHE_COLUMNS = ["signal_date", "code", "dragon_tiger_positive"]
CYQ_CACHE_COLUMNS = ["signal_date", "code", "profit_ratio", "concentration_90"]


def _disable_proxy_env() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)


def _normalize_signal_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _dedupe_join(values: pd.Series, limit: int = 3) -> str:
    items: list[str] = []
    for value in values.astype(str):
        text = value.strip()
        if not text or text.lower() == "nan" or text in items:
            continue
        items.append(text)
        if len(items) >= limit:
            break
    return " | ".join(items)


def _cache_has_requested_dates(cache_df: pd.DataFrame, requested_dates: set[str]) -> bool:
    if cache_df.empty or "signal_date" not in cache_df.columns:
        return False
    cached_dates = set(cache_df["signal_date"].dropna().astype(str))
    return requested_dates.issubset(cached_dates)


def _cache_path_for_code(directory: Path, code: str) -> Path:
    return directory / f"{normalize_code(code)}.csv"


def _market_for_code(code: str) -> str:
    normalized = normalize_code(code)
    if normalized.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return "sh"
    if normalized.startswith(("4", "8", "92")):
        return "bj"
    return "sz"


def _fetch_notice_rows(signal_date: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)

    _disable_proxy_env()
    try:
        raw_df = ak.stock_notice_report(symbol="全部", date=signal_date.replace("-", ""))
    except Exception:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)

    working = raw_df.rename(
        columns={
            "代码": "code",
            "公告标题": "announcement_title",
            "公告类型": "announcement_type",
            "公告日期": "announcement_date",
        }
    ).copy()
    if "code" not in working.columns:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)

    working["code"] = working["code"].map(normalize_code)
    working["signal_date"] = _normalize_signal_date(
        working.get("announcement_date", pd.Series([signal_date] * len(working), index=working.index))
    )
    working["signal_date"] = working["signal_date"].fillna(signal_date)
    working["announcement_title"] = working.get("announcement_title", pd.Series(dtype="object")).map(_safe_text)
    working["announcement_type"] = working.get("announcement_type", pd.Series(dtype="object")).map(_safe_text)
    working = working[working["code"].ne("")].copy()
    working = working[working["signal_date"].eq(signal_date)].copy()
    if working.empty:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)

    grouped = (
        working.groupby(["signal_date", "code"], as_index=False)
        .agg(
            announcement_summary=("announcement_title", _dedupe_join),
            event_summary=("announcement_type", _dedupe_join),
        )
        .copy()
    )
    grouped["event_summary"] = grouped.apply(
        lambda row: " | ".join(text for text in [row.get("event_summary", ""), row.get("announcement_summary", "")] if text),
        axis=1,
    )
    return grouped[NOTICE_CACHE_COLUMNS]


def _ensure_notice_context(signal_dates: list[str]) -> pd.DataFrame:
    if not signal_dates:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)

    requested_dates = {str(date) for date in signal_dates if str(date)}
    cache_df = read_csv_safely(V4_NOTICE_CACHE_CSV)
    if not cache_df.empty:
        cache_df["signal_date"] = _normalize_signal_date(cache_df["signal_date"])
    missing_dates = sorted(date for date in requested_dates if date not in set(cache_df.get("signal_date", pd.Series(dtype="object")).dropna().astype(str)))

    fetched_frames: list[pd.DataFrame] = []
    for signal_date in missing_dates:
        fetched_df = _fetch_notice_rows(signal_date)
        if not fetched_df.empty:
            fetched_frames.append(fetched_df)

    if fetched_frames:
        fetched = pd.concat(fetched_frames, ignore_index=True)
        cache_df = pd.concat([cache_df, fetched], ignore_index=True) if not cache_df.empty else fetched
        cache_df = cache_df.drop_duplicates(["signal_date", "code"], keep="last")
        cache_df = cache_df.sort_values(["signal_date", "code"]).reset_index(drop=True)
        write_csv(cache_df[NOTICE_CACHE_COLUMNS], V4_NOTICE_CACHE_CSV)

    if cache_df.empty:
        return pd.DataFrame(columns=NOTICE_CACHE_COLUMNS)
    return cache_df[cache_df["signal_date"].astype(str).isin(requested_dates)].copy()


def _fetch_fund_flow_rows(code: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception:
        return pd.DataFrame(columns=FUND_FLOW_CACHE_COLUMNS)

    _disable_proxy_env()
    try:
        raw_df = ak.stock_individual_fund_flow(stock=normalize_code(code), market=_market_for_code(code))
    except Exception:
        return pd.DataFrame(columns=FUND_FLOW_CACHE_COLUMNS)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=FUND_FLOW_CACHE_COLUMNS)

    working = raw_df.copy()
    date_column = next((column for column in working.columns if "日期" in str(column)), None)
    flow_column = next((column for column in working.columns if "主力净流入-净占比" in str(column)), None)
    if not date_column or not flow_column:
        return pd.DataFrame(columns=FUND_FLOW_CACHE_COLUMNS)

    result = pd.DataFrame(
        {
            "signal_date": _normalize_signal_date(working[date_column]),
            "code": normalize_code(code),
            "capital_flow_pct": _safe_numeric_series(working[flow_column]),
        }
    )
    result = result.dropna(subset=["signal_date"]).copy()
    result["capital_flow_signal"] = result["capital_flow_pct"]
    return result[FUND_FLOW_CACHE_COLUMNS].drop_duplicates(["signal_date", "code"], keep="last")


def _ensure_code_context(
    codes: list[str],
    requested_dates: list[str],
    directory: Path,
    cache_columns: list[str],
    fetcher,
) -> pd.DataFrame:
    if not codes:
        return pd.DataFrame(columns=cache_columns)

    requested_date_set = {str(date) for date in requested_dates if str(date)}
    frames: list[pd.DataFrame] = []
    for code in sorted({normalize_code(code) for code in codes if normalize_code(code)}):
        cache_path = _cache_path_for_code(directory, code)
        cache_df = read_csv_safely(cache_path)
        if not cache_df.empty and "signal_date" in cache_df.columns:
            cache_df["signal_date"] = _normalize_signal_date(cache_df["signal_date"])
        if not _cache_has_requested_dates(cache_df, requested_date_set):
            fetched_df = fetcher(code)
            if not fetched_df.empty:
                cache_df = fetched_df if cache_df.empty else pd.concat([cache_df, fetched_df], ignore_index=True)
                cache_df = cache_df.drop_duplicates(["signal_date", "code"], keep="last")
                cache_df = cache_df.sort_values(["signal_date", "code"]).reset_index(drop=True)
                write_csv(cache_df[cache_columns], cache_path)
        if not cache_df.empty:
            frames.append(cache_df[cache_df["signal_date"].astype(str).isin(requested_date_set)].copy())

    if not frames:
        return pd.DataFrame(columns=cache_columns)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["signal_date", "code"], keep="last")


def _fetch_lhb_rows(code: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception:
        return pd.DataFrame(columns=LHB_CACHE_COLUMNS)

    _disable_proxy_env()
    try:
        raw_df = ak.stock_lhb_stock_detail_date_em(symbol=normalize_code(code))
    except Exception:
        return pd.DataFrame(columns=LHB_CACHE_COLUMNS)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=LHB_CACHE_COLUMNS)

    working = raw_df.copy()
    date_column = next((column for column in working.columns if "交易日" in str(column) or "日期" in str(column)), None)
    if not date_column:
        return pd.DataFrame(columns=LHB_CACHE_COLUMNS)

    result = pd.DataFrame(
        {
            "signal_date": _normalize_signal_date(working[date_column]),
            "code": normalize_code(code),
            "dragon_tiger_positive": True,
        }
    )
    return result.dropna(subset=["signal_date"])[LHB_CACHE_COLUMNS].drop_duplicates(["signal_date", "code"], keep="last")


def _fetch_cyq_rows(code: str) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception:
        return pd.DataFrame(columns=CYQ_CACHE_COLUMNS)

    _disable_proxy_env()
    raw_df = pd.DataFrame()
    for _ in range(3):
        try:
            raw_df = ak.stock_cyq_em(symbol=normalize_code(code), adjust="qfq")
            if raw_df is not None and not raw_df.empty:
                break
        except Exception:
            raw_df = pd.DataFrame()
            time.sleep(1)
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=CYQ_CACHE_COLUMNS)

    columns = list(raw_df.columns)
    if len(columns) < 6:
        return pd.DataFrame(columns=CYQ_CACHE_COLUMNS)

    result = pd.DataFrame(
        {
            "signal_date": _normalize_signal_date(raw_df.iloc[:, 0]),
            "code": normalize_code(code),
            "profit_ratio": _safe_numeric_series(raw_df.iloc[:, 1]),
            "concentration_90": _safe_numeric_series(raw_df.iloc[:, 5]),
        }
    )
    return result.dropna(subset=["signal_date"])[CYQ_CACHE_COLUMNS].drop_duplicates(["signal_date", "code"], keep="last")


def _ensure_fund_flow_context(codes: list[str], requested_dates: list[str]) -> pd.DataFrame:
    return _ensure_code_context(codes, requested_dates, V4_FUND_FLOW_DIR, FUND_FLOW_CACHE_COLUMNS, _fetch_fund_flow_rows)


def _ensure_lhb_context(codes: list[str], requested_dates: list[str]) -> pd.DataFrame:
    return _ensure_code_context(codes, requested_dates, V4_LHB_DIR, LHB_CACHE_COLUMNS, _fetch_lhb_rows)


def _ensure_cyq_context(codes: list[str], requested_dates: list[str]) -> pd.DataFrame:
    return _ensure_code_context(codes, requested_dates, V4_CYQ_DIR, CYQ_CACHE_COLUMNS, _fetch_cyq_rows)


def enrich_v4_context(feature_df: pd.DataFrame) -> pd.DataFrame:
    if feature_df is None or feature_df.empty:
        return pd.DataFrame()

    working = feature_df.copy()
    if "signal_date" not in working.columns or "code" not in working.columns:
        return working

    working["signal_date"] = _normalize_signal_date(working["signal_date"])
    working["code"] = working["code"].map(normalize_code)
    working = working.dropna(subset=["signal_date"]).copy()
    working = working[working["code"].ne("")].copy()
    if working.empty:
        return feature_df.copy()

    signal_dates = sorted(working["signal_date"].dropna().astype(str).unique().tolist())
    codes = sorted(working["code"].dropna().astype(str).unique().tolist())

    context_frames = [
        _ensure_notice_context(signal_dates),
        _ensure_fund_flow_context(codes, signal_dates),
        _ensure_lhb_context(codes, signal_dates),
        _ensure_cyq_context(codes, signal_dates),
    ]

    enriched = working
    for context_df in context_frames:
        if context_df.empty:
            continue
        enriched = enriched.merge(context_df, how="left", on=["signal_date", "code"])

    if "dragon_tiger_positive" in enriched.columns:
        enriched["dragon_tiger_positive"] = enriched["dragon_tiger_positive"].fillna(False).astype(bool)
    if "capital_flow_pct" in enriched.columns:
        enriched["capital_flow_pct"] = pd.to_numeric(enriched["capital_flow_pct"], errors="coerce")
    if "capital_flow_signal" in enriched.columns:
        enriched["capital_flow_signal"] = pd.to_numeric(enriched["capital_flow_signal"], errors="coerce")
    if "profit_ratio" in enriched.columns:
        enriched["profit_ratio"] = pd.to_numeric(enriched["profit_ratio"], errors="coerce")
    if "concentration_90" in enriched.columns:
        enriched["concentration_90"] = pd.to_numeric(enriched["concentration_90"], errors="coerce")

    if "board_strength" not in enriched.columns:
        enriched["board_strength"] = pd.NA

    return enriched
