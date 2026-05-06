from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from src.paths import backtest_summary_csv_for, followups_csv_for, rule_evaluation_csv_for, signals_csv_for
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, normalize_strategy_version
from src.utils import normalize_code, read_csv_safely, write_csv


BACKTEST_STRONG_THRESHOLD_PCT = 10.0


@dataclass(frozen=True)
class BacktestMetricSpec:
    key: str
    column: str
    settled_column: str | None
    label: str


@dataclass(frozen=True)
class BacktestGroupSpec:
    name: str
    column: str


@dataclass
class BacktestServiceResult:
    strategy_version: str
    generated_at: str
    summary_df: pd.DataFrame
    rule_evaluation_df: pd.DataFrame


BACKTEST_METRIC_SPECS = [
    BacktestMetricSpec("latest", "latest_return_pct", None, "至今收益"),
    BacktestMetricSpec("tail_next_open", "tail_next_open_pct", "settled_tail_next_day", "次日开盘收益"),
    BacktestMetricSpec("tail_next_close", "tail_next_close_pct", "settled_tail_next_day", "次日收盘收益"),
    BacktestMetricSpec("1d", "return_1d_pct", "settled_1d", "1日收益"),
    BacktestMetricSpec("3d", "return_3d_pct", "settled_3d", "3日收益"),
    BacktestMetricSpec("5d", "return_5d_pct", "settled_5d", "5日收益"),
    BacktestMetricSpec("10d", "return_10d_pct", "settled_10d", "10日收益"),
]

BACKTEST_GROUP_SPECS = [
    BacktestGroupSpec("推送层级", "push_level"),
    BacktestGroupSpec("人气排名段", "rank_bucket"),
    BacktestGroupSpec("上榜阶段", "stage_bucket"),
]

RULE_EVALUATION_METRIC_SPECS = [(spec.key, spec.column, spec.settled_column) for spec in BACKTEST_METRIC_SPECS]


def rank_bucket(rank: object) -> str:
    value = pd.to_numeric(pd.Series([rank]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "未知"
    if value <= 3:
        return "Top3"
    if value <= 10:
        return "Top10"
    if value <= 20:
        return "Top20"
    if value <= 50:
        return "Top50"
    return "Top100"


def stage_bucket(consecutive_days: object) -> str:
    value = pd.to_numeric(pd.Series([consecutive_days]), errors="coerce").iloc[0]
    if pd.isna(value) or value <= 1:
        return "首次上榜"
    if value == 2:
        return "连续第2天"
    if value == 3:
        return "连续第3天"
    return "连续4天及以上"


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _safe_group_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "未知"
    text = str(value).strip()
    return text or "未知"


def _prepare_backtest_frame(signal_df: pd.DataFrame, followup_df: pd.DataFrame) -> pd.DataFrame:
    if followup_df.empty:
        return pd.DataFrame()

    df = followup_df.copy()
    if "code" in df.columns:
        df["code"] = df["code"].apply(normalize_code)

    merge_columns = [column for column in ["signal_date", "code", "push_level", "rank", "consecutive_days", "is_pushed"] if column in signal_df.columns]
    if {"signal_date", "code"}.issubset(merge_columns):
        signal_view = signal_df[merge_columns].copy().drop_duplicates(subset=["signal_date", "code"], keep="last")
        signal_view["code"] = signal_view["code"].apply(normalize_code)
        df = df.merge(signal_view, on=["signal_date", "code"], how="left", suffixes=("", "_signal"))
        for column in ["push_level", "rank", "consecutive_days", "is_pushed"]:
            signal_column = f"{column}_signal"
            if signal_column in df.columns:
                if column not in df.columns:
                    df[column] = df[signal_column]
                else:
                    df[column] = df[column].where(df[column].notna(), df[signal_column])
                df = df.drop(columns=[signal_column], errors="ignore")

    if "rank_bucket" not in df.columns:
        df["rank_bucket"] = df.get("rank", pd.Series([None] * len(df))).apply(rank_bucket)
    if "stage_bucket" not in df.columns:
        df["stage_bucket"] = df.get("consecutive_days", pd.Series([None] * len(df))).apply(stage_bucket)
    if "push_level" not in df.columns:
        df["push_level"] = "未知"
    if "is_pushed" not in df.columns:
        df["is_pushed"] = False
    return df


def build_backtest_summary(
    signal_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    generated_at: str | None = None,
) -> pd.DataFrame:
    if signal_df.empty or followup_df.empty:
        return pd.DataFrame()

    normalized_version = normalize_strategy_version(strategy_version)
    generated_at = generated_at or datetime.now().isoformat(timespec="seconds")
    df = _prepare_backtest_frame(signal_df, followup_df)
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for group_spec in BACKTEST_GROUP_SPECS:
        if group_spec.column not in df.columns:
            continue
        grouped = df.groupby(group_spec.column, dropna=False)
        for group_value, group_df in grouped:
            sample_count = int(len(group_df))
            pushed_count = int(_truthy_mask(group_df["is_pushed"]).sum()) if "is_pushed" in group_df.columns else 0
            for metric_spec in BACKTEST_METRIC_SPECS:
                if metric_spec.column not in group_df.columns:
                    continue
                values = pd.to_numeric(group_df[metric_spec.column], errors="coerce")
                if metric_spec.settled_column and metric_spec.settled_column in group_df.columns:
                    values = values[_truthy_mask(group_df[metric_spec.settled_column])]
                values = values.dropna()
                rows.append(
                    {
                        "strategy_version": normalized_version,
                        "generated_at": generated_at,
                        "group_name": group_spec.name,
                        "group_value": _safe_group_value(group_value),
                        "metric_key": metric_spec.key,
                        "metric_label": metric_spec.label,
                        "metric_column": metric_spec.column,
                        "settled_column": metric_spec.settled_column,
                        "sample_count": sample_count,
                        "pushed_count": pushed_count,
                        "valid_count": int(len(values)),
                        "avg_return_pct": round(float(values.mean()), 2) if not values.empty else None,
                        "win_rate_pct": round(float((values > 0).mean() * 100), 2) if not values.empty else None,
                        "strong_rate_pct": round(float((values >= BACKTEST_STRONG_THRESHOLD_PCT).mean() * 100), 2) if not values.empty else None,
                    }
                )
    return pd.DataFrame(rows)


def build_rule_evaluation_view(summary_df: pd.DataFrame, strategy_version: str = DEFAULT_STRATEGY_VERSION) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    normalized_version = normalize_strategy_version(strategy_version)
    key_columns = ["group_name", "group_value", "sample_count", "pushed_count"]
    result = summary_df[key_columns].drop_duplicates().reset_index(drop=True)
    for metric_spec in BACKTEST_METRIC_SPECS:
        metric_df = summary_df[summary_df["metric_key"].eq(metric_spec.key)].copy()
        if metric_df.empty:
            continue
        metric_df = metric_df[
            [
                "group_name",
                "group_value",
                "valid_count",
                "avg_return_pct",
                "win_rate_pct",
                "strong_rate_pct",
            ]
        ].rename(
            columns={
                "valid_count": f"valid_{metric_spec.key}",
                "avg_return_pct": f"avg_{metric_spec.key}",
                "win_rate_pct": f"win_rate_{metric_spec.key}",
                "strong_rate_pct": f"strong_rate_{metric_spec.key}",
            }
        )
        result = result.merge(metric_df, on=["group_name", "group_value"], how="left")
    result["strategy_version"] = normalized_version
    return result


def run_backtest_service(
    signal_df: pd.DataFrame | None = None,
    followup_df: pd.DataFrame | None = None,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
    summary_path: str | None = None,
    rule_evaluation_path: str | None = None,
) -> BacktestServiceResult:
    normalized_version = normalize_strategy_version(strategy_version)
    signal_df = read_csv_safely(signals_csv_for(normalized_version)) if signal_df is None else signal_df.copy()
    followup_df = read_csv_safely(followups_csv_for(normalized_version)) if followup_df is None else followup_df.copy()
    generated_at = datetime.now().isoformat(timespec="seconds")

    summary_df = build_backtest_summary(
        signal_df,
        followup_df,
        strategy_version=normalized_version,
        generated_at=generated_at,
    )
    rule_evaluation_df = build_rule_evaluation_view(summary_df, strategy_version=normalized_version)

    resolved_summary_path = backtest_summary_csv_for(normalized_version) if summary_path is None else summary_path
    resolved_rule_eval_path = rule_evaluation_csv_for(normalized_version) if rule_evaluation_path is None else rule_evaluation_path
    write_csv(summary_df, resolved_summary_path)
    write_csv(rule_evaluation_df, resolved_rule_eval_path)

    return BacktestServiceResult(
        strategy_version=normalized_version,
        generated_at=generated_at,
        summary_df=summary_df,
        rule_evaluation_df=rule_evaluation_df,
    )
