from __future__ import annotations

import pandas as pd

from src.backtest_service import BACKTEST_GROUP_SPECS, BACKTEST_METRIC_SPECS
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, normalize_strategy_version


BACKTEST_METRIC_KEY_TO_LABEL = {str(spec.key): str(spec.label) for spec in BACKTEST_METRIC_SPECS}
BACKTEST_METRIC_LABEL_TO_KEY = {label: key for key, label in BACKTEST_METRIC_KEY_TO_LABEL.items()}
BACKTEST_GROUP_NAMES = [str(spec.name) for spec in BACKTEST_GROUP_SPECS]

BACKTEST_GROUP_ORDER = {
    "推送层级": 0,
    "人气排名段": 1,
    "上榜阶段": 2,
}
BACKTEST_GROUP_VALUE_ORDER = {
    "强推观察": 0,
    "重点观察": 1,
    "普通观察": 2,
    "观察池": 3,
    "不推送": 4,
    "Top3": 10,
    "Top10": 11,
    "Top20": 12,
    "Top50": 13,
    "Top100": 14,
    "首次上榜": 20,
    "连续第2天": 21,
    "连续第3天": 22,
    "连续4天及以上": 23,
}

BACKTEST_SUMMARY_COLUMNS = [
    "strategy_version",
    "generated_at",
    "group_name",
    "group_value",
    "metric_key",
    "metric_label",
    "metric_column",
    "settled_column",
    "sample_count",
    "pushed_count",
    "valid_count",
    "avg_return_pct",
    "win_rate_pct",
    "strong_rate_pct",
]


def normalize_backtest_summary(
    summary_df: pd.DataFrame,
    strategy_version: str | None = None,
) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(columns=BACKTEST_SUMMARY_COLUMNS)

    working = summary_df.copy()
    for column in BACKTEST_SUMMARY_COLUMNS:
        if column not in working.columns:
            working[column] = None

    normalized_version = normalize_strategy_version(strategy_version or DEFAULT_STRATEGY_VERSION)
    if "strategy_version" not in summary_df.columns or working["strategy_version"].isna().all():
        working["strategy_version"] = normalized_version
    else:
        working["strategy_version"] = working["strategy_version"].fillna(normalized_version).astype(str).str.strip().str.lower()

    for column in ["group_name", "group_value", "metric_key", "metric_label", "metric_column", "settled_column", "generated_at"]:
        working[column] = working[column].fillna("").astype(str).str.strip()
    for column in ["sample_count", "pushed_count", "valid_count", "avg_return_pct", "win_rate_pct", "strong_rate_pct"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    order_map = {spec.key: index for index, spec in enumerate(BACKTEST_METRIC_SPECS)}
    working["_group_order"] = working["group_name"].map(BACKTEST_GROUP_ORDER).fillna(99)
    working["_value_order"] = working["group_value"].map(BACKTEST_GROUP_VALUE_ORDER).fillna(99)
    working["_metric_order"] = working["metric_key"].map(order_map).fillna(99)
    working = working.sort_values(
        ["_group_order", "_value_order", "group_name", "group_value", "_metric_order", "metric_key"]
    ).drop(columns=["_group_order", "_value_order", "_metric_order"])
    return working[BACKTEST_SUMMARY_COLUMNS].reset_index(drop=True)


def query_backtest_summary(
    summary_df: pd.DataFrame,
    *,
    strategy_version: str | None = None,
    metric_key: str | None = None,
    group_name: str | None = None,
    group_values: list[str] | None = None,
    min_sample_count: int = 0,
    min_valid_count: int = 0,
) -> pd.DataFrame:
    working = normalize_backtest_summary(summary_df, strategy_version=strategy_version)
    if working.empty:
        return working

    if strategy_version:
        normalized_version = normalize_strategy_version(strategy_version)
        working = working[working["strategy_version"].eq(normalized_version)].copy()
    if metric_key:
        working = working[working["metric_key"].eq(str(metric_key).strip())].copy()
    if group_name:
        working = working[working["group_name"].eq(str(group_name).strip())].copy()
    if group_values:
        normalized_values = {str(value).strip() for value in group_values if str(value).strip()}
        if normalized_values:
            working = working[working["group_value"].isin(normalized_values)].copy()

    working = working[
        pd.to_numeric(working["sample_count"], errors="coerce").fillna(0).ge(max(int(min_sample_count), 0))
    ].copy()
    working = working[
        pd.to_numeric(working["valid_count"], errors="coerce").fillna(0).ge(max(int(min_valid_count), 0))
    ].copy()
    return working.reset_index(drop=True)


def build_backtest_metric_snapshot(
    summary_df: pd.DataFrame,
    *,
    metric_key: str,
    group_name: str,
    strategy_version: str | None = None,
    min_sample_count: int = 0,
    min_valid_count: int = 0,
) -> pd.DataFrame:
    columns = [
        "group_name",
        "group_value",
        "metric_key",
        "metric_label",
        "sample_count",
        "pushed_count",
        "valid_count",
        "avg_return_pct",
        "win_rate_pct",
        "strong_rate_pct",
        "generated_at",
    ]
    working = query_backtest_summary(
        summary_df,
        strategy_version=strategy_version,
        metric_key=metric_key,
        group_name=group_name,
        min_sample_count=min_sample_count,
        min_valid_count=min_valid_count,
    )
    if working.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in working.columns]
    return working[available].reset_index(drop=True)


def build_backtest_metric_matrix(
    summary_df: pd.DataFrame,
    *,
    strategy_version: str | None = None,
    group_name: str | None = None,
    metric_keys: list[str] | None = None,
    min_sample_count: int = 0,
    min_valid_count: int = 0,
) -> pd.DataFrame:
    working = query_backtest_summary(
        summary_df,
        strategy_version=strategy_version,
        group_name=group_name,
        min_sample_count=min_sample_count,
        min_valid_count=min_valid_count,
    )
    if working.empty:
        return pd.DataFrame(columns=["group_name", "group_value", "sample_count", "pushed_count"])

    selected_metric_keys = metric_keys or [spec.key for spec in BACKTEST_METRIC_SPECS]
    base = (
        working[["group_name", "group_value", "sample_count", "pushed_count"]]
        .drop_duplicates(subset=["group_name", "group_value"], keep="last")
        .reset_index(drop=True)
    )
    for metric in selected_metric_keys:
        metric_df = working[working["metric_key"].eq(metric)].copy()
        if metric_df.empty:
            continue
        metric_df = metric_df[
            ["group_name", "group_value", "valid_count", "avg_return_pct", "win_rate_pct", "strong_rate_pct"]
        ].rename(
            columns={
                "valid_count": f"valid_{metric}",
                "avg_return_pct": f"avg_{metric}",
                "win_rate_pct": f"win_rate_{metric}",
                "strong_rate_pct": f"strong_rate_{metric}",
            }
        )
        base = base.merge(metric_df, on=["group_name", "group_value"], how="left")

    base["_group_order"] = base["group_name"].map(BACKTEST_GROUP_ORDER).fillna(99)
    base["_value_order"] = base["group_value"].map(BACKTEST_GROUP_VALUE_ORDER).fillna(99)
    base = base.sort_values(["_group_order", "_value_order", "group_name", "group_value"]).drop(
        columns=["_group_order", "_value_order"]
    )
    return base.reset_index(drop=True)


def build_backtest_compare_table(
    summary_by_version: dict[str, pd.DataFrame],
    *,
    metric_key: str,
    group_name: str,
    min_sample_count: int = 0,
    min_valid_count: int = 0,
) -> pd.DataFrame:
    versions = [
        normalize_strategy_version(version)
        for version, frame in summary_by_version.items()
        if isinstance(frame, pd.DataFrame) and not normalize_backtest_summary(frame, version).empty
    ]
    if not versions:
        return pd.DataFrame(columns=["group_name", "group_value"])

    merged: pd.DataFrame | None = None
    value_columns = ["sample_count", "pushed_count", "valid_count", "avg_return_pct", "win_rate_pct", "strong_rate_pct"]
    for version in versions:
        snapshot_df = build_backtest_metric_snapshot(
            summary_by_version.get(version, pd.DataFrame()),
            metric_key=metric_key,
            group_name=group_name,
            strategy_version=version,
            min_sample_count=min_sample_count,
            min_valid_count=min_valid_count,
        )
        for column in value_columns:
            if column not in snapshot_df.columns:
                snapshot_df[column] = None
        snapshot_df = snapshot_df[["group_name", "group_value", *value_columns]].rename(
            columns={column: f"{version}_{column}" for column in value_columns}
        )
        merged = snapshot_df if merged is None else merged.merge(snapshot_df, on=["group_name", "group_value"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["group_name", "group_value"])

    baseline = versions[0]
    for version in versions[1:]:
        for column in ["pushed_count", "valid_count", "avg_return_pct", "win_rate_pct", "strong_rate_pct"]:
            left = f"{baseline}_{column}"
            right = f"{version}_{column}"
            if {left, right}.issubset(merged.columns):
                merged[f"{version}_vs_{baseline}_{column}_delta"] = (
                    pd.to_numeric(merged[right], errors="coerce") - pd.to_numeric(merged[left], errors="coerce")
                )

    merged["_group_order"] = merged["group_name"].map(BACKTEST_GROUP_ORDER).fillna(99)
    merged["_value_order"] = merged["group_value"].map(BACKTEST_GROUP_VALUE_ORDER).fillna(99)
    merged = merged.sort_values(["_group_order", "_value_order", "group_name", "group_value"]).drop(
        columns=["_group_order", "_value_order"]
    )
    return merged.reset_index(drop=True)
