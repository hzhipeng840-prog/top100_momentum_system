from __future__ import annotations

import pandas as pd


PUSH_LEVEL_ORDER = ["强推观察", "重点观察", "普通观察", "不推送"]
RETURN_METRIC_SPECS = {
    "至今收益": ("latest_return_pct", None),
    "次日开盘收益": ("tail_next_open_pct", "settled_tail_next_day"),
    "次日收盘收益": ("tail_next_close_pct", "settled_tail_next_day"),
    "次日最大冲高": ("tail_next_max_gain_pct", "settled_tail_next_day"),
    "1日收益": ("return_1d_pct", "settled_1d"),
    "3日收益": ("return_3d_pct", "settled_3d"),
    "5日收益": ("return_5d_pct", "settled_5d"),
    "10日收益": ("return_10d_pct", "settled_10d"),
}


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def summarize_push_level_performance(
    followup_df: pd.DataFrame,
    signal_date: str,
    metric_label: str,
) -> pd.DataFrame:
    columns = [
        "signal_date",
        "push_level",
        "sample_count",
        "valid_count",
        "pending_count",
        "up_count",
        "down_or_flat_count",
        "win_rate_pct",
        "avg_return_pct",
        "median_return_pct",
    ]
    if followup_df.empty or metric_label not in RETURN_METRIC_SPECS:
        return pd.DataFrame(columns=columns)

    metric_column, settled_column = RETURN_METRIC_SPECS[metric_label]
    required = {"signal_date", "push_level"}
    if not required.issubset(followup_df.columns) or metric_column not in followup_df.columns:
        return pd.DataFrame(columns=columns)

    working = followup_df.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working = working[working["signal_date"].eq(str(signal_date))].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)

    working["push_level"] = working["push_level"].fillna("未知").astype(str)
    working[metric_column] = pd.to_numeric(working[metric_column], errors="coerce")

    rows: list[dict[str, object]] = []
    push_levels = [level for level in PUSH_LEVEL_ORDER if level in set(working["push_level"])]
    other_levels = sorted(level for level in set(working["push_level"]) if level not in PUSH_LEVEL_ORDER)
    for push_level in push_levels + other_levels:
        group = working[working["push_level"].eq(push_level)].copy()
        sample_count = len(group)
        valid = group[group[metric_column].notna()].copy()
        valid_count = len(valid)

        if settled_column is not None and settled_column in group.columns:
            pending_count = int((~_truthy_mask(group[settled_column])).sum())
        else:
            pending_count = sample_count - valid_count

        up_count = int((valid[metric_column] > 0).sum()) if valid_count else 0
        down_or_flat_count = int((valid[metric_column] <= 0).sum()) if valid_count else 0
        win_rate_pct = round(float((valid[metric_column] > 0).mean() * 100), 2) if valid_count else None
        avg_return_pct = round(float(valid[metric_column].mean()), 2) if valid_count else None
        median_return_pct = round(float(valid[metric_column].median()), 2) if valid_count else None

        rows.append(
            {
                "signal_date": str(signal_date),
                "push_level": push_level,
                "sample_count": sample_count,
                "valid_count": valid_count,
                "pending_count": pending_count,
                "up_count": up_count,
                "down_or_flat_count": down_or_flat_count,
                "win_rate_pct": win_rate_pct,
                "avg_return_pct": avg_return_pct,
                "median_return_pct": median_return_pct,
            }
        )

    result = pd.DataFrame(rows, columns=columns)
    order_map = {level: index for index, level in enumerate(PUSH_LEVEL_ORDER)}
    result["_order"] = result["push_level"].map(order_map).fillna(len(order_map) + 1)
    result = result.sort_values(["_order", "push_level"]).drop(columns=["_order"]).reset_index(drop=True)
    return result


def summarize_push_level_trend(
    followup_df: pd.DataFrame,
    metric_label: str,
    signal_dates: list[str] | None = None,
) -> pd.DataFrame:
    columns = [
        "signal_date",
        "push_level",
        "sample_count",
        "valid_count",
        "pending_count",
        "up_count",
        "down_or_flat_count",
        "win_rate_pct",
        "avg_return_pct",
        "median_return_pct",
    ]
    if followup_df.empty or metric_label not in RETURN_METRIC_SPECS:
        return pd.DataFrame(columns=columns)

    working = followup_df.copy()
    if "signal_date" not in working.columns:
        return pd.DataFrame(columns=columns)
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    available_dates = sorted(date for date in working["signal_date"].dropna().astype(str).unique() if date)
    if not available_dates:
        return pd.DataFrame(columns=columns)

    if signal_dates:
        selected_dates = [str(date) for date in signal_dates if str(date) in set(available_dates)]
    else:
        selected_dates = available_dates
    if not selected_dates:
        return pd.DataFrame(columns=columns)

    frames = [
        summarize_push_level_performance(working, signal_date, metric_label)
        for signal_date in selected_dates
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=columns)

    trend_df = pd.concat(frames, ignore_index=True)
    trend_df["signal_date"] = pd.to_datetime(trend_df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    order_map = {level: index for index, level in enumerate(PUSH_LEVEL_ORDER)}
    trend_df["_order"] = trend_df["push_level"].map(order_map).fillna(len(order_map) + 1)
    trend_df = trend_df.sort_values(["signal_date", "_order", "push_level"]).drop(columns=["_order"]).reset_index(drop=True)
    return trend_df[columns]
