from __future__ import annotations

import pandas as pd

from src.market_data import load_price_data
from src.paths import FEATURES_CSV, INTRADAY_SNAPSHOT_CSV, RAW_POPULARITY_CSV
from src.strategy_profiles import (
    DEFAULT_STRATEGY_VERSION,
    normalize_strategy_version,
    strategy_capture_priority,
)
from src.utils import normalize_code, parse_number, pct_change, read_csv_safely, write_csv


FEATURE_COLUMNS = [
    "signal_date",
    "code",
    "name",
    "rank",
    "popularity_score",
    "source",
    "capture_type",
    "snapshot_time",
    "first_seen_date",
    "appearance_count",
    "consecutive_days",
    "previous_rank",
    "rank_change",
    "price_status",
    "price_date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "day_return_pct",
    "pre3_return_pct",
    "pre5_return_pct",
    "close_position",
    "volume_ratio_5",
    "dist_ma5_pct",
    "dist_ma10_pct",
    "dist_ma20_pct",
    "dist_high60_pct",
    "upper_shadow_pct",
    "limit_up_like",
    "one_word_like",
    "price_lag_days",
]

INTRADAY_SNAPSHOT_COLUMNS = [
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


def load_popularity() -> pd.DataFrame:
    df = read_csv_safely(RAW_POPULARITY_CSV)
    if df.empty:
        return pd.DataFrame(
            columns=["signal_date", "rank", "code", "name", "popularity_score", "source", "capture_type", "snapshot_time"]
        )
    df = df.copy()
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["code"] = df["code"].apply(normalize_code)
    if "capture_type" not in df.columns:
        df["capture_type"] = "post_close"
    else:
        df["capture_type"] = df["capture_type"].fillna("").astype(str).str.strip().replace("", "post_close")
    if "snapshot_time" not in df.columns:
        df["snapshot_time"] = df["signal_date"] + " 15:00:00"
    else:
        df["snapshot_time"] = df["snapshot_time"].fillna("").astype(str).str.strip()
        df.loc[df["snapshot_time"].eq(""), "snapshot_time"] = df["signal_date"] + " 15:00:00"
    df = df.dropna(subset=["signal_date", "rank"])
    df = df[df["code"].ne("")]
    return df.sort_values(["signal_date", "snapshot_time", "rank"], na_position="last").reset_index(drop=True)


def load_intraday_snapshots() -> pd.DataFrame:
    df = read_csv_safely(INTRADAY_SNAPSHOT_CSV)
    if df.empty:
        return pd.DataFrame(columns=INTRADAY_SNAPSHOT_COLUMNS + ["signal_date"])

    working = df.copy()
    for column in INTRADAY_SNAPSHOT_COLUMNS:
        if column not in working.columns:
            working[column] = None
    working["code"] = working["code"].apply(normalize_code)
    working["capture_type"] = working["capture_type"].fillna("").astype(str).str.strip()
    working["snapshot_time"] = working["snapshot_time"].fillna("").astype(str).str.strip()
    working["signal_date"] = pd.to_datetime(working["snapshot_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in [
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
    ]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["signal_date"])
    working = working[working["code"].ne("")]
    return working.sort_values(["signal_date", "snapshot_time", "code"]).reset_index(drop=True)


def select_strategy_snapshots(
    popularity_df: pd.DataFrame,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    if popularity_df.empty:
        return popularity_df

    priority = strategy_capture_priority(strategy_version)
    df = popularity_df.copy()
    selected_groups: list[pd.DataFrame] = []
    for _, day_df in df.groupby("signal_date", sort=True):
        source_df = pd.DataFrame()
        for capture_type in priority:
            matched = day_df[day_df["capture_type"].astype(str).eq(capture_type)].copy()
            if not matched.empty:
                source_df = matched
                break
        if source_df.empty:
            source_df = day_df.copy()

        snapshot_time = source_df["snapshot_time"].dropna().astype(str).max()
        if snapshot_time:
            selected = source_df[source_df["snapshot_time"].astype(str).eq(snapshot_time)].copy()
        else:
            selected = source_df.copy()
        selected = selected.sort_values("rank", na_position="last").drop_duplicates("code", keep="first")
        selected_groups.append(selected)

    if not selected_groups:
        return pd.DataFrame(columns=popularity_df.columns)
    return pd.concat(selected_groups, ignore_index=True).sort_values(["signal_date", "rank"], na_position="last").reset_index(drop=True)


def select_daily_main_snapshots(popularity_df: pd.DataFrame) -> pd.DataFrame:
    return select_strategy_snapshots(popularity_df, strategy_version=DEFAULT_STRATEGY_VERSION)


def add_appearance_stats(popularity_df: pd.DataFrame) -> pd.DataFrame:
    if popularity_df.empty:
        return popularity_df

    df = popularity_df.copy()
    df["signal_ts"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.normalize()
    unique_dates = sorted(df["signal_ts"].dropna().unique())
    date_to_index = {pd.Timestamp(date): index for index, date in enumerate(unique_dates)}

    state: dict[str, dict] = {}
    rows: list[dict] = []
    for _, row in df.sort_values(["signal_ts", "snapshot_time", "rank"], na_position="last").iterrows():
        code = normalize_code(row.get("code"))
        current_date = pd.Timestamp(row.get("signal_ts"))
        current_index = date_to_index.get(current_date)
        previous = state.get(code, {})

        previous_count = int(previous.get("appearance_count", 0))
        previous_date_index = previous.get("date_index")
        previous_consecutive = int(previous.get("consecutive_days", 0))
        if previous_date_index == current_index:
            appearance_count = previous_count
            consecutive_days = previous_consecutive
        else:
            appearance_count = previous_count + 1
            consecutive_days = previous_consecutive + 1 if previous_date_index == current_index - 1 else 1
        first_seen_date = previous.get("first_seen_date") or current_date.strftime("%Y-%m-%d")
        previous_rank = previous.get("rank")
        rank = parse_number(row.get("rank"))
        rank_change = None
        if previous_rank is not None and rank is not None:
            rank_change = round(float(previous_rank) - rank, 2)

        enriched = row.to_dict()
        enriched.update(
            {
                "first_seen_date": first_seen_date,
                "appearance_count": appearance_count,
                "consecutive_days": consecutive_days,
                "previous_rank": previous_rank,
                "rank_change": rank_change,
            }
        )
        rows.append(enriched)

        state[code] = {
            "appearance_count": appearance_count,
            "consecutive_days": consecutive_days,
            "date_index": current_index,
            "rank": rank,
            "first_seen_date": first_seen_date,
        }

    result = pd.DataFrame(rows)
    return result.drop(columns=["signal_ts"], errors="ignore")


def _rolling_mean(series: pd.Series, window: int) -> float | None:
    values = pd.to_numeric(series.tail(window), errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _feature_for_price_date(price_df: pd.DataFrame, signal_date: str) -> dict:
    target_date = pd.Timestamp(signal_date).normalize()
    history_df = price_df[price_df["date"] <= target_date].copy()
    if history_df.empty:
        return {"price_status": "no_price_before_signal"}

    current = history_df.iloc[-1]
    current_date = pd.Timestamp(current.get("date")).normalize()
    current_open = parse_number(current.get("open"))
    current_close = parse_number(current.get("close"))
    current_high = parse_number(current.get("high"))
    current_low = parse_number(current.get("low"))
    current_volume = parse_number(current.get("volume"))

    previous_rows = history_df.iloc[:-1]
    previous_close = parse_number(previous_rows.iloc[-1].get("close")) if not previous_rows.empty else None
    pre3_close = parse_number(history_df.iloc[-4].get("close")) if len(history_df) >= 4 else None
    pre5_close = parse_number(history_df.iloc[-6].get("close")) if len(history_df) >= 6 else None

    close_position = None
    if current_high is not None and current_low is not None and current_close is not None:
        if current_high > current_low:
            close_position = round((current_close - current_low) / (current_high - current_low), 3)
        elif abs(current_close - current_high) <= max(current_close * 0.001, 0.01):
            close_position = 1.0

    volume_ratio_5 = None
    if current_volume is not None and "volume" in previous_rows.columns:
        volume_mean = pd.to_numeric(previous_rows["volume"].tail(5), errors="coerce").dropna().mean()
        if pd.notna(volume_mean) and volume_mean > 0:
            volume_ratio_5 = round(current_volume / volume_mean, 2)

    dist_values: dict[str, float | None] = {}
    for window in (5, 10, 20):
        ma_value = _rolling_mean(history_df["close"], window)
        dist_values[f"dist_ma{window}_pct"] = pct_change(current_close, ma_value)

    high60 = pd.to_numeric(history_df["high"].tail(60), errors="coerce").dropna().max()
    dist_high60_pct = pct_change(current_close, float(high60)) if pd.notna(high60) and high60 > 0 else None

    day_return_pct = pct_change(current_close, previous_close)
    limit_up_like = bool(day_return_pct is not None and day_return_pct >= 9.5)
    one_word_like = False
    if limit_up_like and current_high is not None and current_low is not None and current_close is not None:
        one_word_like = abs(current_high - current_low) <= max(current_close * 0.001, 0.01)

    upper_shadow_pct = None
    if current_high is not None and current_low is not None and current_open is not None and current_close is not None:
        if current_high > current_low:
            upper_shadow_pct = round((current_high - max(current_open, current_close)) / (current_high - current_low), 3)
        elif one_word_like:
            upper_shadow_pct = 0.0

    return {
        "price_status": "ok",
        "price_date": current_date.strftime("%Y-%m-%d"),
        "open": current_open,
        "close": current_close,
        "high": current_high,
        "low": current_low,
        "volume": current_volume,
        "day_return_pct": day_return_pct,
        "pre3_return_pct": pct_change(current_close, pre3_close),
        "pre5_return_pct": pct_change(current_close, pre5_close),
        "close_position": close_position,
        "volume_ratio_5": volume_ratio_5,
        "dist_ma5_pct": dist_values["dist_ma5_pct"],
        "dist_ma10_pct": dist_values["dist_ma10_pct"],
        "dist_ma20_pct": dist_values["dist_ma20_pct"],
        "dist_high60_pct": dist_high60_pct,
        "upper_shadow_pct": upper_shadow_pct,
        "limit_up_like": limit_up_like,
        "one_word_like": one_word_like,
        "price_lag_days": int((target_date - current_date).days),
    }


def _match_intraday_snapshot(
    snapshot_df: pd.DataFrame,
    code: str,
    signal_date: str,
    capture_type: str,
    snapshot_time: str,
) -> pd.Series | None:
    if snapshot_df.empty:
        return None

    matched = snapshot_df[snapshot_df["code"].astype(str).eq(code)].copy()
    if matched.empty:
        return None

    if capture_type:
        captured = matched[matched["capture_type"].astype(str).eq(str(capture_type))].copy()
        if not captured.empty:
            matched = captured

    if snapshot_time:
        exact = matched[matched["snapshot_time"].astype(str).eq(str(snapshot_time))].copy()
        if not exact.empty:
            matched = exact
        else:
            same_day = matched[matched["signal_date"].astype(str).eq(str(signal_date))].copy()
            if not same_day.empty:
                matched = same_day
    else:
        same_day = matched[matched["signal_date"].astype(str).eq(str(signal_date))].copy()
        if not same_day.empty:
            matched = same_day

    if matched.empty:
        return None
    return matched.sort_values("snapshot_time").iloc[-1]


def _feature_for_intraday_snapshot(
    price_df: pd.DataFrame,
    snapshot_row: pd.Series,
    signal_date: str,
) -> dict:
    target_date = pd.Timestamp(signal_date).normalize()
    history_df = price_df[price_df["date"] < target_date].copy()

    current_close = parse_number(snapshot_row.get("last_price"))
    if current_close is None:
        return {"price_status": "intraday_snapshot_missing_price"}

    current_open = parse_number(snapshot_row.get("open"))
    current_high = parse_number(snapshot_row.get("day_high_so_far"))
    current_low = parse_number(snapshot_row.get("day_low_so_far"))
    current_volume = parse_number(snapshot_row.get("volume_so_far"))
    previous_close = parse_number(snapshot_row.get("prev_close"))
    if previous_close is None and not history_df.empty:
        previous_close = parse_number(history_df.iloc[-1].get("close"))

    if current_high is None:
        current_high = current_close
    if current_low is None:
        current_low = current_close
    if current_open is None:
        current_open = previous_close if previous_close is not None else current_close

    pre3_close = parse_number(history_df.iloc[-3].get("close")) if len(history_df) >= 3 else None
    pre5_close = parse_number(history_df.iloc[-5].get("close")) if len(history_df) >= 5 else None

    close_position = None
    if current_high is not None and current_low is not None:
        if current_high > current_low:
            close_position = round((current_close - current_low) / (current_high - current_low), 3)
        elif abs(current_close - current_high) <= max(current_close * 0.001, 0.01):
            close_position = 1.0

    volume_ratio_5 = None
    if current_volume is not None and "volume" in history_df.columns:
        volume_mean = pd.to_numeric(history_df["volume"].tail(5), errors="coerce").dropna().mean()
        if pd.notna(volume_mean) and volume_mean > 0:
            volume_ratio_5 = round(current_volume / volume_mean, 2)

    close_series = pd.concat(
        [
            pd.to_numeric(history_df.get("close", pd.Series(dtype=float)), errors="coerce").dropna(),
            pd.Series([current_close], dtype=float),
        ],
        ignore_index=True,
    )
    dist_values: dict[str, float | None] = {}
    for window in (5, 10, 20):
        ma_value = _rolling_mean(close_series, window)
        dist_values[f"dist_ma{window}_pct"] = pct_change(current_close, ma_value)

    high_series = pd.concat(
        [
            pd.to_numeric(history_df.get("high", pd.Series(dtype=float)), errors="coerce").dropna(),
            pd.Series([current_high], dtype=float),
        ],
        ignore_index=True,
    )
    high60 = pd.to_numeric(high_series.tail(60), errors="coerce").dropna().max()
    dist_high60_pct = pct_change(current_close, float(high60)) if pd.notna(high60) and high60 > 0 else None

    day_return_pct = parse_number(snapshot_row.get("current_return_pct"))
    if day_return_pct is None:
        day_return_pct = pct_change(current_close, previous_close)
    limit_up_like = bool(day_return_pct is not None and day_return_pct >= 9.5)
    one_word_like = False
    if limit_up_like and current_high is not None and current_low is not None:
        one_word_like = abs(current_high - current_low) <= max(current_close * 0.001, 0.01)

    upper_shadow_pct = None
    if current_high is not None and current_low is not None and current_open is not None:
        if current_high > current_low:
            upper_shadow_pct = round((current_high - max(current_open, current_close)) / (current_high - current_low), 3)
        elif one_word_like:
            upper_shadow_pct = 0.0

    return {
        "price_status": "ok",
        "price_date": target_date.strftime("%Y-%m-%d"),
        "open": current_open,
        "close": current_close,
        "high": current_high,
        "low": current_low,
        "volume": current_volume,
        "day_return_pct": day_return_pct,
        "pre3_return_pct": pct_change(current_close, pre3_close),
        "pre5_return_pct": pct_change(current_close, pre5_close),
        "close_position": close_position,
        "volume_ratio_5": volume_ratio_5,
        "dist_ma5_pct": dist_values["dist_ma5_pct"],
        "dist_ma10_pct": dist_values["dist_ma10_pct"],
        "dist_ma20_pct": dist_values["dist_ma20_pct"],
        "dist_high60_pct": dist_high60_pct,
        "upper_shadow_pct": upper_shadow_pct,
        "limit_up_like": limit_up_like,
        "one_word_like": one_word_like,
        "price_lag_days": 0,
    }


def _feature_for_signal_row(
    row: pd.Series,
    price_df: pd.DataFrame,
    intraday_snapshot_df: pd.DataFrame,
) -> dict:
    capture_type = str(row.get("capture_type") or "").strip()
    if capture_type.startswith("intraday_"):
        snapshot_row = _match_intraday_snapshot(
            intraday_snapshot_df,
            code=normalize_code(row.get("code")),
            signal_date=str(row.get("signal_date") or ""),
            capture_type=capture_type,
            snapshot_time=str(row.get("snapshot_time") or ""),
        )
        if snapshot_row is not None:
            intraday_features = _feature_for_intraday_snapshot(
                price_df=price_df,
                snapshot_row=snapshot_row,
                signal_date=str(row.get("signal_date")),
            )
            if intraday_features.get("price_status") == "ok":
                return intraday_features

    if price_df.empty:
        return {"price_status": "missing_price_file"}
    return _feature_for_price_date(price_df, str(row.get("signal_date")))


def build_daily_features(
    popularity_df: pd.DataFrame | None = None,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    strategy_version = normalize_strategy_version(strategy_version)
    base_df = load_popularity() if popularity_df is None else popularity_df.copy()
    if base_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    base_df = select_strategy_snapshots(base_df, strategy_version=strategy_version)
    base_df = add_appearance_stats(base_df)
    use_intraday_features = base_df["capture_type"].fillna("").astype(str).str.startswith("intraday_").any()
    intraday_snapshot_df = load_intraday_snapshots() if use_intraday_features else pd.DataFrame()
    price_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    for _, row in base_df.iterrows():
        code = normalize_code(row.get("code"))
        if code not in price_cache:
            price_cache[code] = load_price_data(code)
        rows.append(
            {
                **row.to_dict(),
                **_feature_for_signal_row(
                    row=row,
                    price_df=price_cache[code],
                    intraday_snapshot_df=intraday_snapshot_df,
                ),
            }
        )

    feature_df = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        if column not in feature_df.columns:
            feature_df[column] = None
    return feature_df[FEATURE_COLUMNS].sort_values(["signal_date", "rank"], na_position="last").reset_index(drop=True)


def save_daily_features(feature_df: pd.DataFrame) -> None:
    write_csv(feature_df, FEATURES_CSV)
