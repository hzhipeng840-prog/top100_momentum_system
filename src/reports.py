from __future__ import annotations

import pandas as pd

from src.backtest_service import build_backtest_summary, build_rule_evaluation_view, run_backtest_service
from src.paths import (
    backtest_summary_csv_for,
    fast_strategy_audit_csv_for,
    fast_strategy_csv_for,
    fast_strategy_history_csv_for,
    followups_csv_for,
    latest_push_csv_for,
    lesson_evaluation_csv_for,
    rule_evaluation_csv_for,
    signals_csv_for,
    strong_recap_csv_for,
)
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, normalize_strategy_version
from src.trading_calendar import latest_expected_market_date
from src.utils import read_csv_safely, write_csv


MARKET_AUDIT_COLUMNS = [
    "market_price_date",
    "market_lag_days",
    "market_source",
]


FAST_STRATEGY_COLUMNS = [
    "strategy_date",
    "training_date",
    "analysis_window",
    "learned_rule",
    "fast_score",
    "fast_level",
    "next_session_plan",
    "fit_reasons",
    "rank",
    "code",
    "name",
    "push_level",
    "emotion_score",
    "day_return_pct",
    "pre5_return_pct",
    "market_regime",
    "market_1d_pct",
    "market_5d_pct",
    *MARKET_AUDIT_COLUMNS,
    "relative_1d_pct",
    "relative_5d_pct",
    "market_adjustment",
    "close_position",
    "volume_ratio_5",
    "dist_ma20_pct",
    "consecutive_days",
    "rank_change",
    "capture_type",
    "snapshot_time",
    "reasons",
    "risks",
]

FAST_STRATEGY_AUDIT_COLUMNS = [
    "strategy_date",
    "training_date",
    "audit_status",
    "audit_scope",
    "audit_result",
    "lesson_type",
    "lesson_note",
    "rank",
    "code",
    "name",
    "push_level",
    "fast_level",
    "fast_score",
    *MARKET_AUDIT_COLUMNS,
    "observed_days",
    "latest_return_pct",
    "tail_next_open_pct",
    "tail_next_close_pct",
    "tail_next_max_gain_pct",
    "return_1d_pct",
    "return_3d_pct",
    "return_5d_pct",
    "max_gain_5d_pct",
    "capture_type",
    "snapshot_time",
    "reasons",
    "risks",
]

RULE_EVAL_METRIC_SPECS = [
    ("latest", "latest_return_pct", None),
    ("tail_next_open", "tail_next_open_pct", "settled_tail_next_day"),
    ("tail_next_close", "tail_next_close_pct", "settled_tail_next_day"),
    ("1d", "return_1d_pct", "settled_1d"),
    ("3d", "return_3d_pct", "settled_3d"),
    ("5d", "return_5d_pct", "settled_5d"),
    ("10d", "return_10d_pct", "settled_10d"),
]


def _best_return_metric_columns(strategy_version: str) -> list[str]:
    columns = [
        "return_5d_pct",
        "return_3d_pct",
        "latest_return_pct",
        "max_gain_5d_pct",
        "max_gain_3d_pct",
    ]
    if strategy_version == "v3":
        return [
            "tail_next_close_pct",
            "tail_next_max_gain_pct",
            "tail_next_open_pct",
            *columns,
        ]
    return columns


def _rank_bucket(rank: object) -> str:
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


def _stage_bucket(consecutive_days: object) -> str:
    value = pd.to_numeric(pd.Series([consecutive_days]), errors="coerce").iloc[0]
    if pd.isna(value) or value <= 1:
        return "首次上榜"
    if value == 2:
        return "连续第2天"
    if value == 3:
        return "连续第3天"
    return "连续4天及以上"


def _latest_signal_slice(signal_df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    if signal_df.empty or "signal_date" not in signal_df.columns:
        return None, pd.DataFrame()
    latest_date = signal_df["signal_date"].dropna().astype(str).max()
    latest_df = signal_df[signal_df["signal_date"].astype(str).eq(latest_date)].copy()
    if latest_df.empty:
        return latest_date, latest_df
    if "snapshot_time" in latest_df.columns:
        latest_snapshot = latest_df["snapshot_time"].dropna().astype(str).max()
        if latest_snapshot:
            latest_df = latest_df[latest_df["snapshot_time"].astype(str).eq(latest_snapshot)].copy()
    return latest_date, latest_df


def build_latest_push(signal_df: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    latest_date, latest_df = _latest_signal_slice(signal_df)
    if latest_date is None:
        return pd.DataFrame()
    latest_df = latest_df[latest_df["is_pushed"].astype(str).str.lower().isin(["true", "1"])]
    if latest_df.empty:
        return pd.DataFrame()
    latest_df["emotion_score"] = pd.to_numeric(latest_df["emotion_score"], errors="coerce")
    latest_df["rank"] = pd.to_numeric(latest_df["rank"], errors="coerce")
    keep = [
        "signal_date",
        "rank",
        "code",
        "name",
        "push_level",
        "emotion_score",
        "day_return_pct",
        "pre5_return_pct",
        "market_regime",
        "market_1d_pct",
        "market_5d_pct",
        *MARKET_AUDIT_COLUMNS,
        "relative_1d_pct",
        "relative_5d_pct",
        "close_position",
        "volume_ratio_5",
        "dist_ma20_pct",
        "consecutive_days",
        "rank_change",
        "capture_type",
        "snapshot_time",
        "reasons",
        "risks",
        "suggested_action",
    ]
    available = [column for column in keep if column in latest_df.columns]
    result = latest_df.sort_values(["emotion_score", "rank"], ascending=[False, True])[available]
    if limit is not None and limit > 0:
        return result.head(limit)
    return result


def build_strong_recap(
    followup_df: pd.DataFrame,
    threshold_pct: float = 15,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    if followup_df.empty:
        return pd.DataFrame()

    strategy_version = normalize_strategy_version(strategy_version)
    df = followup_df.copy()
    return_columns = [column for column in df.columns if column.startswith("return_") and column.endswith("d_pct")]
    gain_columns = [column for column in df.columns if column.startswith("max_gain_") and column.endswith("d_pct")]
    metric_columns = return_columns + gain_columns + ["latest_return_pct"]
    if strategy_version == "v3":
        metric_columns = ["tail_next_open_pct", "tail_next_close_pct", "tail_next_max_gain_pct", *metric_columns]
    for column in metric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    available_metrics = [column for column in metric_columns if column in df.columns]
    if not available_metrics:
        return pd.DataFrame()
    df["best_return_available"] = df[available_metrics].max(axis=1, skipna=True)
    df = df[df["best_return_available"] >= threshold_pct].copy()
    if df.empty:
        return pd.DataFrame()

    keep = [
        "signal_date",
        "code",
        "name",
        "rank",
        "push_level",
        "emotion_score",
        "observed_days",
        "tail_next_open_pct",
        "tail_next_close_pct",
        "tail_next_max_gain_pct",
        "latest_return_pct",
        "best_return_available",
        "return_3d_pct",
        "return_5d_pct",
        "return_10d_pct",
        "max_gain_3d_pct",
        "max_gain_5d_pct",
        "max_gain_10d_pct",
        "reasons",
        "risks",
    ]
    available = [column for column in keep if column in df.columns]
    return df.sort_values("best_return_available", ascending=False)[available].reset_index(drop=True)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _format_rule_values(values: list[str]) -> str:
    clean = [str(value) for value in values if str(value) and str(value) != "nan"]
    return "、".join(clean) if clean else "不限"


def _top_reason_terms(series: pd.Series, limit: int = 4) -> str:
    counts: dict[str, int] = {}
    for text in series.dropna().astype(str):
        for term in text.split("；"):
            term = term.strip()
            if not term or term == "-":
                continue
            counts[term] = counts.get(term, 0) + 1
    if not counts:
        return "原因待补充"
    terms = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return "、".join(term for term, _ in terms)


def _choose_fast_training_date(signal_df: pd.DataFrame, followup_df: pd.DataFrame, latest_date: str) -> str | None:
    dates = sorted(pd.to_datetime(signal_df["signal_date"], errors="coerce").dropna().dt.normalize().unique())
    dates = [pd.Timestamp(date) for date in dates if pd.Timestamp(date).strftime("%Y-%m-%d") < latest_date]
    if not dates:
        return None

    latest_ts = pd.Timestamp(latest_date)
    preferred = latest_ts - pd.Timedelta(days=7)
    if preferred in dates:
        return preferred.strftime("%Y-%m-%d")

    scored: list[tuple[int, int, float, pd.Timestamp]] = []
    followup = followup_df.copy()
    followup["signal_date"] = pd.to_datetime(followup["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for date in dates:
        date_text = date.strftime("%Y-%m-%d")
        group = followup[followup["signal_date"].eq(date_text)]
        valid_5d = int(_numeric_column(group, "return_5d_pct").notna().sum())
        valid_3d = int(_numeric_column(group, "return_3d_pct").notna().sum())
        observed = float(_numeric_column(group, "observed_days").max()) if not group.empty else 0.0
        scored.append((valid_5d, valid_3d, observed, date))
    if not scored:
        return dates[-1].strftime("%Y-%m-%d")
    best = max(scored, key=lambda item: (item[0], item[1], item[2], item[3]))
    return best[3].strftime("%Y-%m-%d")


def _with_strategy_return(df: pd.DataFrame, strategy_version: str = DEFAULT_STRATEGY_VERSION) -> pd.DataFrame:
    result = df.copy()
    strategy_version = normalize_strategy_version(strategy_version)
    metric_columns = _best_return_metric_columns(strategy_version)
    available = [column for column in metric_columns if column in result.columns]
    if not available:
        result["strategy_return_pct"] = None
        return result
    for column in available:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["strategy_return_pct"] = result[available].max(axis=1, skipna=True)
    return result


def _select_strong_groups(train_df: pd.DataFrame, column: str, strong_threshold: float) -> list[str]:
    if column not in train_df.columns or train_df.empty:
        return []
    frame = train_df[[column, "strategy_return_pct"]].dropna(subset=["strategy_return_pct"]).copy()
    if frame.empty:
        return []
    overall_avg = float(frame["strategy_return_pct"].mean())
    overall_strong_rate = float((frame["strategy_return_pct"] >= strong_threshold).mean() * 100)
    rows: list[dict] = []
    for value, group in frame.groupby(column, dropna=False):
        returns = pd.to_numeric(group["strategy_return_pct"], errors="coerce").dropna()
        if returns.empty:
            continue
        rows.append(
            {
                "value": str(value),
                "sample_count": len(returns),
                "avg_return": float(returns.mean()),
                "strong_rate": float((returns >= strong_threshold).mean() * 100),
            }
        )
    stats = pd.DataFrame(rows)
    if stats.empty:
        return []
    selected = stats[
        (stats["sample_count"] >= 2)
        & (stats["avg_return"] >= overall_avg)
        & (stats["strong_rate"] >= overall_strong_rate)
    ].copy()
    if selected.empty:
        selected = stats.sort_values(["strong_rate", "avg_return", "sample_count"], ascending=[False, False, False]).head(1)
    else:
        selected = selected.sort_values(["strong_rate", "avg_return", "sample_count"], ascending=[False, False, False]).head(3)
    return selected["value"].astype(str).tolist()


def _learn_fast_rule(train_df: pd.DataFrame, strategy_version: str = DEFAULT_STRATEGY_VERSION) -> dict:
    strategy_version = normalize_strategy_version(strategy_version)
    train_df = _with_strategy_return(train_df, strategy_version=strategy_version)
    returns = pd.to_numeric(train_df["strategy_return_pct"], errors="coerce").dropna()
    if returns.empty:
        return {
            "strong_threshold": 15.0,
            "push_levels": [],
            "rank_buckets": [],
            "stage_buckets": [],
            "score_floor": 60.0,
            "pre5_low": None,
            "pre5_high": None,
            "volume_low": None,
            "volume_high": None,
            "close_floor": None,
            "summary": "近样本尚未形成有效收益结算，暂用原始情绪分排序。",
        }

    strong_threshold = 15.0 if int((returns >= 15).sum()) >= 3 else 10.0
    winners = train_df[pd.to_numeric(train_df["strategy_return_pct"], errors="coerce") >= strong_threshold].copy()
    if winners.empty:
        winners = train_df.sort_values("strategy_return_pct", ascending=False).head(10).copy()

    score_values = _numeric_column(winners, "emotion_score").dropna()
    pre5_values = _numeric_column(winners, "pre5_return_pct").dropna()
    volume_values = _numeric_column(winners, "volume_ratio_5").dropna()
    close_values = _numeric_column(winners, "close_position").dropna()

    score_floor = max(50.0, round(float(score_values.quantile(0.25)), 2)) if not score_values.empty else 60.0
    pre5_low = round(float(pre5_values.quantile(0.15)), 2) if not pre5_values.empty else None
    pre5_high = round(float(pre5_values.quantile(0.9)), 2) if not pre5_values.empty else None
    volume_low = round(float(volume_values.quantile(0.15)), 2) if not volume_values.empty else None
    volume_high = round(float(volume_values.quantile(0.9)), 2) if not volume_values.empty else None
    close_floor = round(float(close_values.quantile(0.25)), 3) if not close_values.empty else None

    push_levels = _select_strong_groups(train_df, "push_level", strong_threshold)
    rank_buckets = _select_strong_groups(train_df, "rank_bucket", strong_threshold)
    stage_buckets = _select_strong_groups(train_df, "stage_bucket", strong_threshold)
    low_level_winners = winners[winners["push_level"].astype(str).isin(["普通观察", "不推送"])].copy() if "push_level" in winners.columns else pd.DataFrame()
    low_level_lesson = ""
    if not low_level_winners.empty:
        notes: list[str] = []
        for level, group in low_level_winners.groupby("push_level", dropna=False):
            notes.append(f"{level}{len(group)}只: {_top_reason_terms(group.get('reasons', pd.Series(dtype='object')))}")
        low_level_lesson = "；".join(notes)
    summary = (
        f"近样本强势阈值{strong_threshold:.0f}%，"
        f"偏好推送层级={_format_rule_values(push_levels)}，"
        f"排名段={_format_rule_values(rank_buckets)}，"
        f"上榜阶段={_format_rule_values(stage_buckets)}，"
        f"情绪分不低于{score_floor:.0f}。"
    )
    if low_level_lesson:
        summary += f" 低层级强势反哺：{low_level_lesson}。"

    return {
        "strong_threshold": strong_threshold,
        "push_levels": push_levels,
        "rank_buckets": rank_buckets,
        "stage_buckets": stage_buckets,
        "score_floor": score_floor,
        "pre5_low": pre5_low,
        "pre5_high": pre5_high,
        "volume_low": volume_low,
        "volume_high": volume_high,
        "close_floor": close_floor,
        "low_level_lesson": low_level_lesson,
        "summary": summary,
    }


def _numeric_value(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _fast_thresholds_for_market(market_regime: object) -> tuple[float, float, float, float]:
    if str(market_regime or "") == "弱势":
        return 90.0, 80.0, 70.0, 75.0
    return 85.0, 75.0, 65.0, 65.0


def _market_adjustment(row: pd.Series, reasons: list[str]) -> float:
    market_regime = str(row.get("market_regime") or "未知")
    relative_1d = _numeric_value(row.get("relative_1d_pct"))
    relative_5d = _numeric_value(row.get("relative_5d_pct"))
    adjustment = 0.0

    if market_regime == "弱势":
        adjustment -= 8.0
        if (relative_1d is not None and relative_1d >= 3.0) or (relative_5d is not None and relative_5d >= 5.0):
            adjustment += 10.0
            reasons.append("弱市仍跑赢大盘")
        else:
            adjustment -= 6.0
            reasons.append("弱市相对强度不足")
    elif market_regime == "震荡":
        if (relative_1d is not None and relative_1d >= 2.0) or (relative_5d is not None and relative_5d >= 4.0):
            adjustment += 4.0
            reasons.append("相对大盘偏强")
    elif market_regime == "强势":
        if relative_1d is not None and relative_1d < 0:
            adjustment -= 3.0
            reasons.append("强市跑输大盘")

    return adjustment


def _dominant_market_regime(df: pd.DataFrame) -> str:
    if df.empty or "market_regime" not in df.columns:
        return "未知"
    values = df["market_regime"].dropna().astype(str)
    if values.empty:
        return "未知"
    return str(values.mode().iloc[0])


def _score_fast_candidate(row: pd.Series, rule: dict) -> dict:
    score = 0.0
    reasons: list[str] = []
    emotion_score = pd.to_numeric(pd.Series([row.get("emotion_score")]), errors="coerce").iloc[0]
    if pd.notna(emotion_score):
        score += min(float(emotion_score), 100.0) * 0.45

    if str(row.get("push_level")) in rule["push_levels"]:
        score += 15
        reasons.append("命中近样本强势层级")
    if str(row.get("rank_bucket")) in rule["rank_buckets"]:
        score += 12
        reasons.append("命中近样本强势排名段")
    if str(row.get("stage_bucket")) in rule["stage_buckets"]:
        score += 12
        reasons.append("命中近样本强势上榜阶段")
    if pd.notna(emotion_score) and float(emotion_score) >= float(rule["score_floor"]):
        score += 10
        reasons.append("情绪分达近期强势线")

    pre5 = pd.to_numeric(pd.Series([row.get("pre5_return_pct")]), errors="coerce").iloc[0]
    if rule["pre5_low"] is not None and rule["pre5_high"] is not None and pd.notna(pre5):
        if float(rule["pre5_low"]) <= float(pre5) <= float(rule["pre5_high"]):
            score += 8
            reasons.append("近5日涨幅落在近期胜出区间")

    volume = pd.to_numeric(pd.Series([row.get("volume_ratio_5")]), errors="coerce").iloc[0]
    if rule["volume_low"] is not None and rule["volume_high"] is not None and pd.notna(volume):
        if float(rule["volume_low"]) <= float(volume) <= float(rule["volume_high"]):
            score += 8
            reasons.append("量比落在近期胜出区间")

    close_position = pd.to_numeric(pd.Series([row.get("close_position")]), errors="coerce").iloc[0]
    if rule["close_floor"] is not None and pd.notna(close_position) and float(close_position) >= float(rule["close_floor"]):
        score += 8
        reasons.append("收盘承接不弱于近期胜出样本")

    risks = str(row.get("risks") or "")
    if any(text in risks for text in ["一字", "过热", "过猛", "上影线"]):
        score -= 6
        reasons.append("风险项扣分")

    market_adjustment = _market_adjustment(row, reasons)
    score += market_adjustment

    score = round(max(score, 0.0), 2)
    main_line, priority_line, backup_line, _ = _fast_thresholds_for_market(row.get("market_regime"))
    if score >= main_line:
        fast_level = "下个交易日主盯"
        plan = "先看开盘分歧后的承接，强转弱或量能失控就放弃。"
    elif score >= priority_line:
        fast_level = "优先观察"
        plan = "放入第一观察池，等待回落承接或人气继续上升。"
    elif score >= backup_line:
        fast_level = "备选观察"
        plan = "只在板块和人气共振时考虑，避免追高。"
    else:
        fast_level = "只看不追"
        plan = "当前只做跟踪样本，等下一次滚动复盘验证。"

    return {
        "fast_score": score,
        "fast_level": fast_level,
        "next_session_plan": plan,
        "fit_reasons": "；".join(reasons) if reasons else "-",
        "market_adjustment": round(market_adjustment, 2),
    }


def build_fast_strategy(
    signal_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    if signal_df.empty or followup_df.empty or "signal_date" not in signal_df.columns:
        return pd.DataFrame(columns=FAST_STRATEGY_COLUMNS)

    strategy_version = normalize_strategy_version(strategy_version)
    signal_df = signal_df.copy()
    followup_df = followup_df.copy()
    signal_df["signal_date"] = pd.to_datetime(signal_df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    followup_df["signal_date"] = pd.to_datetime(followup_df["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    latest_date, latest_df = _latest_signal_slice(signal_df)
    if not latest_date or latest_df.empty:
        return pd.DataFrame(columns=FAST_STRATEGY_COLUMNS)
    training_date = _choose_fast_training_date(signal_df, followup_df, latest_date)
    if not training_date:
        return pd.DataFrame(columns=FAST_STRATEGY_COLUMNS)

    signal_cols = [
        "signal_date",
        "code",
        "name",
        "rank",
        "push_level",
        "emotion_score",
        "day_return_pct",
        "pre5_return_pct",
        "close_position",
        "volume_ratio_5",
        "dist_ma20_pct",
        "consecutive_days",
        "rank_change",
        "capture_type",
        "snapshot_time",
        "reasons",
        "risks",
    ]
    available_signal_cols = [column for column in signal_cols if column in signal_df.columns]
    train_signals = signal_df[signal_df["signal_date"].eq(training_date)][available_signal_cols].copy()
    if "snapshot_time" in train_signals.columns and not train_signals.empty:
        latest_training_snapshot = train_signals["snapshot_time"].dropna().astype(str).max()
        if latest_training_snapshot:
            train_signals = train_signals[train_signals["snapshot_time"].astype(str).eq(latest_training_snapshot)].copy()
    train_followups = followup_df[followup_df["signal_date"].eq(training_date)].copy()
    train_df = train_signals.merge(train_followups, on=["signal_date", "code"], how="left", suffixes=("", "_followup"))
    if "rank_bucket" not in train_df.columns:
        train_df["rank_bucket"] = train_df["rank"].apply(_rank_bucket)
    if "stage_bucket" not in train_df.columns:
        train_df["stage_bucket"] = train_df["consecutive_days"].apply(_stage_bucket)
    rule = _learn_fast_rule(train_df, strategy_version=strategy_version)

    latest_df = latest_df.copy()
    latest_df["rank_bucket"] = latest_df["rank"].apply(_rank_bucket)
    latest_df["stage_bucket"] = latest_df["consecutive_days"].apply(_stage_bucket)
    scored = pd.DataFrame([_score_fast_candidate(row, rule) for _, row in latest_df.iterrows()])
    latest_df = pd.concat([latest_df.reset_index(drop=True), scored], axis=1)
    latest_df["strategy_date"] = latest_date
    latest_df["training_date"] = training_date
    latest_df["analysis_window"] = f"{training_date} -> {latest_date}"
    latest_df["learned_rule"] = rule["summary"]
    latest_market_regime = _dominant_market_regime(latest_df)
    if latest_market_regime == "弱势":
        latest_df["learned_rule"] = latest_df["learned_rule"] + " 当前为弱市模式：入池门槛提高，只优先保留明显跑赢大盘的票。"

    latest_df["fast_score"] = pd.to_numeric(latest_df["fast_score"], errors="coerce")
    latest_df["rank"] = pd.to_numeric(latest_df["rank"], errors="coerce")
    _, _, _, selected_floor = _fast_thresholds_for_market(latest_market_regime)
    selected = latest_df[latest_df["fast_score"] >= selected_floor].copy()
    if selected.empty:
        selected = latest_df.sort_values(["fast_score", "emotion_score", "rank"], ascending=[False, False, True]).head(20).copy()
    else:
        selected = selected.sort_values(["fast_score", "emotion_score", "rank"], ascending=[False, False, True]).head(30).copy()

    for column in FAST_STRATEGY_COLUMNS:
        if column not in selected.columns:
            selected[column] = None
    return selected[FAST_STRATEGY_COLUMNS].reset_index(drop=True)


FAST_STRATEGY_LOCK_COLUMNS = ["strategy_date", "training_date", "capture_type", "snapshot_time"]
FAST_STRATEGY_MARKET_CONTEXT_COLUMNS = [
    "market_regime",
    "market_1d_pct",
    "market_5d_pct",
    *MARKET_AUDIT_COLUMNS,
    "relative_1d_pct",
    "relative_5d_pct",
]


def _fill_missing_market_context(base: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    return _fill_missing_market_context_by_keys(base, source, FAST_STRATEGY_LOCK_COLUMNS + ["code"])


def _fill_missing_market_context_by_keys(base: pd.DataFrame, source: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if base.empty or source.empty or not set(keys).issubset(base.columns) or not set(keys).issubset(source.columns):
        return base

    available_context = [column for column in FAST_STRATEGY_MARKET_CONTEXT_COLUMNS if column in source.columns]
    if not available_context:
        return base

    result = base.copy()
    context = source[keys + available_context].drop_duplicates(keys, keep="last")
    merged = result.merge(context, how="left", on=keys, suffixes=("", "_source"))
    for column in available_context:
        source_column = f"{column}_source"
        if column not in merged.columns or source_column not in merged.columns:
            continue
        merged[column] = merged[column].astype("object")
        current_text = merged[column].astype(str).str.strip().str.lower()
        missing_mask = merged[column].isna() | current_text.isin(["", "nan", "none", "nat"])
        merged.loc[missing_mask, column] = merged.loc[missing_mask, source_column]
    return merged.drop(columns=[f"{column}_source" for column in available_context], errors="ignore")


def _normalize_fast_strategy_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in FAST_STRATEGY_COLUMNS:
        if column not in result.columns:
            result[column] = None
    for column in FAST_STRATEGY_LOCK_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result["code"] = result["code"].astype(str).str.zfill(6)
    result["fast_score"] = pd.to_numeric(result["fast_score"], errors="coerce")
    result["rank"] = pd.to_numeric(result["rank"], errors="coerce")
    return result[FAST_STRATEGY_COLUMNS].copy()


def _locked_current_fast_strategy(incoming: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    if incoming.empty or history.empty:
        return incoming
    incoming_keys = incoming[FAST_STRATEGY_LOCK_COLUMNS].drop_duplicates()
    locked = history.merge(incoming_keys, how="inner", on=FAST_STRATEGY_LOCK_COLUMNS)
    if locked.empty:
        return incoming
    locked = _normalize_fast_strategy_frame(locked)
    locked = _fill_missing_market_context(locked, _normalize_fast_strategy_frame(incoming))
    return locked.sort_values(["fast_score", "rank"], ascending=[False, True], na_position="last").reset_index(drop=True)


def _signal_snapshot_keys(signal_df: pd.DataFrame) -> pd.DataFrame:
    required = ["signal_date", "capture_type", "snapshot_time"]
    if signal_df.empty or not set(required).issubset(signal_df.columns):
        return pd.DataFrame(columns=["strategy_date", "capture_type", "snapshot_time"])
    keys = signal_df[required].copy()
    keys["strategy_date"] = pd.to_datetime(keys["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    keys["capture_type"] = keys["capture_type"].fillna("").astype(str)
    keys["snapshot_time"] = keys["snapshot_time"].fillna("").astype(str)
    return keys[["strategy_date", "capture_type", "snapshot_time"]].dropna(subset=["strategy_date"]).drop_duplicates()


def _signal_market_context(signal_df: pd.DataFrame) -> pd.DataFrame:
    signal_context_keys = ["strategy_date", "capture_type", "snapshot_time", "code"]
    required = ["signal_date", "capture_type", "snapshot_time", "code"]
    if signal_df.empty or not set(required).issubset(signal_df.columns):
        return pd.DataFrame()

    available_context = [column for column in FAST_STRATEGY_MARKET_CONTEXT_COLUMNS if column in signal_df.columns]
    if not available_context:
        return pd.DataFrame()

    context = signal_df[required + available_context].copy()
    context["strategy_date"] = pd.to_datetime(context["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    context["capture_type"] = context["capture_type"].fillna("").astype(str)
    context["snapshot_time"] = context["snapshot_time"].fillna("").astype(str)
    context["code"] = context["code"].astype(str).str.zfill(6)
    return (
        context[["strategy_date", "capture_type", "snapshot_time", "code"] + available_context]
        .dropna(subset=["strategy_date"])
        .drop_duplicates(signal_context_keys, keep="last")
        .reset_index(drop=True)
    )


def _filter_history_to_signal_snapshots(history_df: pd.DataFrame, signal_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    keys = _signal_snapshot_keys(signal_df)
    if keys.empty:
        return history_df
    history = history_df.copy()
    for column in ["strategy_date", "capture_type", "snapshot_time"]:
        if column not in history.columns:
            history[column] = None
        history[column] = history[column].fillna("").astype(str)
    return history.merge(keys.assign(_main_snapshot=True), how="inner", on=["strategy_date", "capture_type", "snapshot_time"]).drop(columns=["_main_snapshot"])


def update_fast_strategy_history(
    fast_strategy_df: pd.DataFrame,
    signal_df: pd.DataFrame | None = None,
    history_path=None,
) -> pd.DataFrame:
    history_path = history_path or fast_strategy_history_csv_for(DEFAULT_STRATEGY_VERSION)
    existing = read_csv_safely(history_path)
    signal_context = _signal_market_context(signal_df) if signal_df is not None else pd.DataFrame()
    if signal_df is not None and not existing.empty:
        existing = _filter_history_to_signal_snapshots(existing, signal_df)
    if fast_strategy_df.empty:
        if existing.empty:
            return pd.DataFrame(columns=FAST_STRATEGY_COLUMNS)
        existing = _normalize_fast_strategy_frame(existing)
        if not signal_context.empty:
            existing = _fill_missing_market_context_by_keys(existing, signal_context, ["strategy_date", "capture_type", "snapshot_time", "code"])
            write_csv(existing, history_path)
        return existing

    incoming = _normalize_fast_strategy_frame(fast_strategy_df)
    incoming_full = incoming.copy()
    if not existing.empty:
        existing = _normalize_fast_strategy_frame(existing)
        existing = _fill_missing_market_context(existing, incoming_full)
        if not signal_context.empty:
            existing = _fill_missing_market_context_by_keys(existing, signal_context, ["strategy_date", "capture_type", "snapshot_time", "code"])
        existing_keys = existing[FAST_STRATEGY_LOCK_COLUMNS].drop_duplicates()
        incoming = incoming.merge(
            existing_keys.assign(_already_locked=True),
            how="left",
            on=FAST_STRATEGY_LOCK_COLUMNS,
        )
        incoming = incoming[incoming["_already_locked"].isna()].drop(columns=["_already_locked"])
    combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
    combined = _normalize_fast_strategy_frame(combined)
    if not signal_context.empty:
        combined = _fill_missing_market_context_by_keys(combined, signal_context, ["strategy_date", "capture_type", "snapshot_time", "code"])
    combined = combined.drop_duplicates(["strategy_date", "training_date", "capture_type", "snapshot_time", "code"], keep="last")
    combined = combined.sort_values(["strategy_date", "fast_score", "rank"], ascending=[True, False, True], na_position="last")
    combined = combined[FAST_STRATEGY_COLUMNS].reset_index(drop=True)
    write_csv(combined, history_path)
    return combined


def _add_best_return(df: pd.DataFrame, strategy_version: str = DEFAULT_STRATEGY_VERSION) -> pd.DataFrame:
    result = df.copy()
    strategy_version = normalize_strategy_version(strategy_version)
    metric_columns = _best_return_metric_columns(strategy_version)
    available = [column for column in metric_columns if column in result.columns]
    for column in available:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if available:
        result["best_return_available"] = result[available].max(axis=1, skipna=True)
    else:
        result["best_return_available"] = None
    return result


def _audit_status(row: pd.Series) -> str:
    observed = pd.to_numeric(pd.Series([row.get("observed_days")]), errors="coerce").iloc[0]
    if pd.isna(observed) or float(observed) <= 0:
        strategy_date = pd.to_datetime(row.get("strategy_date"), errors="coerce")
        expected_date = latest_expected_market_date().tz_localize(None)
        if pd.notna(strategy_date) and pd.Timestamp(strategy_date).normalize() < expected_date.normalize():
            return "待补行情"
        return "等待结算"
    if pd.notna(pd.to_numeric(pd.Series([row.get("tail_next_close_pct")]), errors="coerce").iloc[0]):
        return "次日已审"
    if pd.notna(pd.to_numeric(pd.Series([row.get("return_5d_pct")]), errors="coerce").iloc[0]):
        return "5日已审"
    if pd.notna(pd.to_numeric(pd.Series([row.get("return_3d_pct")]), errors="coerce").iloc[0]):
        return "3日已审"
    return "1日已审"


def _audit_fast_candidate(row: pd.Series, strong_threshold: float) -> dict:
    status = _audit_status(row)
    best_return = pd.to_numeric(pd.Series([row.get("best_return_available")]), errors="coerce").iloc[0]
    latest_return = pd.to_numeric(pd.Series([row.get("latest_return_pct")]), errors="coerce").iloc[0]
    fast_level = str(row.get("fast_level") or "")
    is_priority = fast_level in {"下个交易日主盯", "优先观察"}

    if status == "待补行情":
        return {
            "audit_result": "待补行情",
            "lesson_type": "pending_data",
            "lesson_note": "信号日后应已有行情，但本地日K还没补齐；补齐后会自动重新审查。",
        }
    if status == "等待结算":
        return {
            "audit_result": "等待下个交易日",
            "lesson_type": "pending",
            "lesson_note": "已记录本次策略，等下个交易日行情更新后自动结算。",
        }
    if pd.notna(best_return) and float(best_return) >= strong_threshold:
        return {
            "audit_result": "优先成功" if is_priority else "备选走强",
            "lesson_type": "priority_success" if is_priority else "backup_success",
            "lesson_note": f"{fast_level} 后续最高表现达到 {float(best_return):.2f}%，保留这类命中特征。",
        }
    if is_priority and pd.notna(latest_return) and float(latest_return) <= 0:
        return {
            "audit_result": "优先走弱",
            "lesson_type": "priority_failed",
            "lesson_note": f"{fast_level} 未兑现，复盘风险项：{row.get('risks', '-')}",
        }
    return {
        "audit_result": "暂未强势",
        "lesson_type": "neutral",
        "lesson_note": "继续等待更多交易日或降低权重。",
    }


def build_fast_strategy_audit(
    history_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    strong_threshold: float = 15,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    if history_df.empty or signal_df.empty or followup_df.empty:
        return pd.DataFrame(columns=FAST_STRATEGY_AUDIT_COLUMNS)

    strategy_version = normalize_strategy_version(strategy_version)
    history = history_df.copy()
    signals = signal_df.copy()
    followups = followup_df.copy()
    for frame, date_column in [(history, "strategy_date"), (signals, "signal_date"), (followups, "signal_date")]:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
        if "code" in frame.columns:
            frame["code"] = frame["code"].astype(str).str.zfill(6)

    followup_cols = [
        "signal_date",
        "code",
        "observed_days",
        "latest_return_pct",
        "tail_next_open_pct",
        "tail_next_close_pct",
        "tail_next_max_gain_pct",
        "return_1d_pct",
        "return_3d_pct",
        "return_5d_pct",
        "max_gain_5d_pct",
    ]
    followup_available = [column for column in followup_cols if column in followups.columns]
    audited = history.merge(
        followups[followup_available],
        left_on=["strategy_date", "code"],
        right_on=["signal_date", "code"],
        how="left",
    ).drop(columns=["signal_date"], errors="ignore")
    audited = _add_best_return(audited, strategy_version=strategy_version)
    audited["audit_scope"] = "快策略候选"
    audited["audit_status"] = audited.apply(_audit_status, axis=1)
    fast_notes = pd.DataFrame([_audit_fast_candidate(row, strong_threshold) for _, row in audited.iterrows()])
    audited = pd.concat([audited.reset_index(drop=True), fast_notes], axis=1)

    strategy_dates = sorted(history["strategy_date"].dropna().astype(str).unique())
    low_level_rows: list[pd.DataFrame] = []
    selected_keys = set(zip(history["strategy_date"].astype(str), history["code"].astype(str)))
    for strategy_date in strategy_dates:
        day_signals = signals[signals["signal_date"].eq(strategy_date)].copy()
        day_signals = day_signals[day_signals["push_level"].astype(str).isin(["普通观察", "不推送"])]
        if day_signals.empty:
            continue
        merged = day_signals.merge(
            followups[followup_available],
            on=["signal_date", "code"],
            how="left",
        )
        merged = _add_best_return(merged, strategy_version=strategy_version)
        merged = merged[pd.to_numeric(merged["best_return_available"], errors="coerce") >= strong_threshold].copy()
        if merged.empty:
            continue
        merged["strategy_date"] = strategy_date
        merged["training_date"] = None
        merged["fast_level"] = [
            "已入快策略" if (strategy_date, code) in selected_keys else "未入快策略"
            for code in merged["code"].astype(str)
        ]
        merged["fast_score"] = None
        merged["audit_scope"] = "低层级强势反哺"
        merged["audit_status"] = merged.apply(_audit_status, axis=1)
        merged["audit_result"] = "普通/不推送走强"
        merged["lesson_type"] = "low_level_winner"
        merged["lesson_note"] = merged.apply(
            lambda row: f"{row.get('push_level', '-')} 后来走强，原因反哺：{_top_reason_terms(pd.Series([row.get('reasons')]), limit=3)}",
            axis=1,
        )
        low_level_rows.append(merged)

    if low_level_rows:
        audited = pd.concat([audited, *low_level_rows], ignore_index=True)

    for column in FAST_STRATEGY_AUDIT_COLUMNS:
        if column not in audited.columns:
            audited[column] = None
    audited["observed_days"] = pd.to_numeric(audited["observed_days"], errors="coerce")
    audited["fast_score"] = pd.to_numeric(audited["fast_score"], errors="coerce")
    audited["rank"] = pd.to_numeric(audited["rank"], errors="coerce")
    audited["latest_return_pct"] = pd.to_numeric(audited["latest_return_pct"], errors="coerce")
    scope_order = {"快策略候选": 0, "低层级强势反哺": 1}
    fast_level_order = {"下个交易日主盯": 0, "优先观察": 1, "备选观察": 2, "只看不追": 3, "未入快策略": 4}
    audit_result_order = {
        "等待下个交易日": 0,
        "待补行情": 1,
        "优先成功": 2,
        "优先走弱": 3,
        "暂未强势": 4,
        "备选走强": 5,
        "普通/不推送走强": 6,
    }
    audited["_scope_order"] = audited["audit_scope"].map(scope_order).fillna(9)
    audited["_fast_level_order"] = audited["fast_level"].map(fast_level_order).fillna(8)
    audited["_audit_result_order"] = audited["audit_result"].map(audit_result_order).fillna(9)
    audited = audited.sort_values(
        ["strategy_date", "_scope_order", "_fast_level_order", "_audit_result_order", "fast_score", "latest_return_pct", "rank"],
        ascending=[False, True, True, True, False, False, True],
        na_position="last",
    )
    audited = audited.drop(columns=["_scope_order", "_fast_level_order", "_audit_result_order"], errors="ignore")
    return audited[FAST_STRATEGY_AUDIT_COLUMNS].reset_index(drop=True)


def _summarize_group(df: pd.DataFrame, group_name: str, group_value: str) -> dict:
    result = {
        "group_name": group_name,
        "group_value": group_value,
        "sample_count": len(df),
        "pushed_count": int(df["is_pushed"].astype(str).str.lower().isin(["true", "1"]).sum()) if "is_pushed" in df.columns else 0,
    }
    for metric_key, column, settled_column in RULE_EVAL_METRIC_SPECS:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if settled_column in df.columns:
            settled_mask = df[settled_column].astype(str).str.lower().isin(["true", "1"])
            values = values[settled_mask]
        values = values.dropna()
        result[f"valid_{metric_key}"] = len(values)
        result[f"avg_{metric_key}"] = round(float(values.mean()), 2) if not values.empty else None
        result[f"win_rate_{metric_key}"] = round(float((values > 0).mean() * 100), 2) if not values.empty else None
        result[f"strong_rate_{metric_key}"] = round(float((values >= 10).mean() * 100), 2) if not values.empty else None
    return result


def _legacy_build_rule_evaluation(
    signal_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    if signal_df.empty or followup_df.empty:
        return pd.DataFrame()

    strategy_version = normalize_strategy_version(strategy_version)
    df = followup_df.copy()
    if "rank_bucket" not in df.columns:
        df["rank_bucket"] = df["rank"].apply(_rank_bucket)
    if "stage_bucket" not in df.columns:
        signal_stage = signal_df[["signal_date", "code", "consecutive_days"]].copy() if {"signal_date", "code", "consecutive_days"}.issubset(signal_df.columns) else pd.DataFrame()
        if not signal_stage.empty:
            df = df.merge(signal_stage, on=["signal_date", "code"], how="left", suffixes=("", "_signal"))
        df["stage_bucket"] = df.get("consecutive_days", pd.Series([None] * len(df))).apply(_stage_bucket)

    rows: list[dict] = []
    for group_name, column in [("推送层级", "push_level"), ("人气排名段", "rank_bucket"), ("上榜阶段", "stage_bucket")]:
        if column not in df.columns:
            continue
        for value, group in df.groupby(column, dropna=False):
            rows.append(_summarize_group(group, group_name, str(value)))
    result = pd.DataFrame(rows)
    if not result.empty:
        result["strategy_version"] = strategy_version
    return result


def build_rule_evaluation(
    signal_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    if signal_df.empty or followup_df.empty:
        return pd.DataFrame()
    summary_df = build_backtest_summary(
        signal_df=signal_df,
        followup_df=followup_df,
        strategy_version=strategy_version,
    )
    return build_rule_evaluation_view(summary_df, strategy_version=strategy_version)


LESSON_EVALUATION_COLUMNS = [
    "lesson_type",
    "sample_count",
    "valid_latest",
    "avg_latest_return_pct",
    "win_rate_latest_pct",
    "valid_3d",
    "avg_3d_return_pct",
    "win_rate_3d_pct",
    "valid_5d",
    "avg_5d_return_pct",
    "win_rate_5d_pct",
    "valid_max_gain_5d",
    "avg_max_gain_5d_pct",
    "strong_rate_5d_pct",
]


def _summarize_lesson_group(df: pd.DataFrame, lesson_type: str, strong_threshold: float) -> dict:
    result = {
        "lesson_type": lesson_type,
        "sample_count": len(df),
    }

    metric_specs = [
        ("latest_return_pct", "valid_latest", "avg_latest_return_pct", "win_rate_latest_pct"),
        ("return_3d_pct", "valid_3d", "avg_3d_return_pct", "win_rate_3d_pct"),
        ("return_5d_pct", "valid_5d", "avg_5d_return_pct", "win_rate_5d_pct"),
        ("max_gain_5d_pct", "valid_max_gain_5d", "avg_max_gain_5d_pct", None),
    ]
    for source_column, valid_key, avg_key, win_key in metric_specs:
        if source_column not in df.columns:
            continue
        values = pd.to_numeric(df[source_column], errors="coerce").dropna()
        result[valid_key] = len(values)
        result[avg_key] = round(float(values.mean()), 2) if not values.empty else None
        if win_key is not None:
            result[win_key] = round(float((values > 0).mean() * 100), 2) if not values.empty else None

    max_gain_values = pd.to_numeric(df.get("max_gain_5d_pct"), errors="coerce").dropna() if "max_gain_5d_pct" in df.columns else pd.Series(dtype="float64")
    result["strong_rate_5d_pct"] = round(float((max_gain_values >= strong_threshold).mean() * 100), 2) if not max_gain_values.empty else None
    return result


def build_lesson_evaluation(audit_df: pd.DataFrame, strong_threshold: float = 15) -> pd.DataFrame:
    if audit_df.empty or "lesson_type" not in audit_df.columns:
        return pd.DataFrame(columns=LESSON_EVALUATION_COLUMNS)

    df = audit_df.copy()
    df["lesson_type"] = df["lesson_type"].fillna("").astype(str)
    df = df[df["lesson_type"].ne("")]
    if df.empty:
        return pd.DataFrame(columns=LESSON_EVALUATION_COLUMNS)

    rows = [_summarize_lesson_group(group, str(lesson_type), strong_threshold) for lesson_type, group in df.groupby("lesson_type", dropna=False)]
    result = pd.DataFrame(rows)
    for column in LESSON_EVALUATION_COLUMNS:
        if column not in result.columns:
            result[column] = None
    result = result[LESSON_EVALUATION_COLUMNS]
    result["sample_count"] = pd.to_numeric(result["sample_count"], errors="coerce")
    result = result.sort_values(["sample_count", "strong_rate_5d_pct", "avg_5d_return_pct"], ascending=[False, False, False], na_position="last")
    return result.reset_index(drop=True)


def build_reports(
    signal_df: pd.DataFrame | None = None,
    followup_df: pd.DataFrame | None = None,
    latest_push_limit: int | None = None,
    strong_return_threshold_pct: float = 15,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> dict:
    strategy_version = normalize_strategy_version(strategy_version)
    signal_df = read_csv_safely(signals_csv_for(strategy_version)) if signal_df is None else signal_df.copy()
    followup_df = read_csv_safely(followups_csv_for(strategy_version)) if followup_df is None else followup_df.copy()

    latest_push_path = latest_push_csv_for(strategy_version)
    fast_strategy_path = fast_strategy_csv_for(strategy_version)
    fast_strategy_history_path = fast_strategy_history_csv_for(strategy_version)
    fast_strategy_audit_path = fast_strategy_audit_csv_for(strategy_version)
    strong_recap_path = strong_recap_csv_for(strategy_version)
    rule_evaluation_path = rule_evaluation_csv_for(strategy_version)
    lesson_evaluation_path = lesson_evaluation_csv_for(strategy_version)
    backtest_summary_path = backtest_summary_csv_for(strategy_version)

    latest_push_df = build_latest_push(signal_df, limit=latest_push_limit)
    fast_strategy_df = build_fast_strategy(signal_df, followup_df, strategy_version=strategy_version)
    fast_strategy_history_df = update_fast_strategy_history(
        fast_strategy_df,
        signal_df=signal_df,
        history_path=fast_strategy_history_path,
    )
    fast_strategy_df = _locked_current_fast_strategy(fast_strategy_df, fast_strategy_history_df)
    fast_strategy_audit_df = build_fast_strategy_audit(
        history_df=fast_strategy_history_df,
        signal_df=signal_df,
        followup_df=followup_df,
        strong_threshold=strong_return_threshold_pct,
        strategy_version=strategy_version,
    )
    strong_recap_df = build_strong_recap(
        followup_df,
        threshold_pct=strong_return_threshold_pct,
        strategy_version=strategy_version,
    )
    backtest_result = run_backtest_service(
        signal_df=signal_df,
        followup_df=followup_df,
        strategy_version=strategy_version,
        summary_path=backtest_summary_path,
        rule_evaluation_path=rule_evaluation_path,
    )
    rule_eval_df = backtest_result.rule_evaluation_df
    lesson_eval_df = build_lesson_evaluation(fast_strategy_audit_df, strong_threshold=strong_return_threshold_pct)

    for frame in [latest_push_df, fast_strategy_df, fast_strategy_audit_df, strong_recap_df, rule_eval_df, lesson_eval_df]:
        if not frame.empty or "strategy_version" not in frame.columns:
            frame["strategy_version"] = strategy_version

    write_csv(latest_push_df, latest_push_path)
    write_csv(fast_strategy_df, fast_strategy_path)
    write_csv(fast_strategy_audit_df, fast_strategy_audit_path)
    write_csv(strong_recap_df, strong_recap_path)
    write_csv(lesson_eval_df, lesson_evaluation_path)

    return {
        "strategy_version": strategy_version,
        "latest_push_rows": len(latest_push_df),
        "fast_strategy_rows": len(fast_strategy_df),
        "fast_strategy_history_rows": len(fast_strategy_history_df),
        "fast_strategy_audit_rows": len(fast_strategy_audit_df),
        "strong_recap_rows": len(strong_recap_df),
        "rule_evaluation_rows": len(rule_eval_df),
        "backtest_summary_rows": len(backtest_result.summary_df),
        "lesson_evaluation_rows": len(lesson_eval_df),
        "latest_push_path": str(latest_push_path),
        "fast_strategy_path": str(fast_strategy_path),
        "fast_strategy_history_path": str(fast_strategy_history_path),
        "fast_strategy_audit_path": str(fast_strategy_audit_path),
        "strong_recap_path": str(strong_recap_path),
        "rule_evaluation_path": str(rule_evaluation_path),
        "backtest_summary_path": str(backtest_summary_path),
        "lesson_evaluation_path": str(lesson_evaluation_path),
    }
