from __future__ import annotations

import pandas as pd

from src.signals import OBSERVATION_POOL_LEVEL
from src.strategy_profiles import normalize_strategy_version
from src.utils import normalize_code


V3_HEALTH_COOLDOWN_SCORE = 102.0
V3_HEALTH_WINDOW = 10
V3_HEALTH_MIN_VALID = 10
V3_HEALTH_WIN_RATE_FLOOR = 45.0
V3_HEALTH_AVG_RETURN_FLOOR = 0.0
V3_HEALTH_LOSS5_CEILING = 25.0
V3_HEALTH_RISK_NOTE = "v3健康度冷却，强推票暂停执行"
V3_HEALTH_ACTION = "v3健康度冷却：近期强推表现偏弱，暂停执行，只留观察。"


def _truthy_mask(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _append_note(value: object, note: str) -> str:
    text = str(value or "").strip()
    if text in {"", "-", "nan", "None"}:
        return note
    if note in text:
        return text
    return f"{text}；{note}"


def _add_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "signal_date" in result.columns:
        result["_health_signal_date"] = pd.to_datetime(result["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        result["_health_signal_date"] = None
    if "code" in result.columns:
        result["_health_code"] = result["code"].astype(str).map(normalize_code)
    else:
        result["_health_code"] = ""
    for column in ["capture_type", "snapshot_time"]:
        if column in result.columns:
            result[f"_health_{column}"] = result[column].fillna("").astype(str)
        else:
            result[f"_health_{column}"] = ""
    return result


def _health_key_columns(signal_df: pd.DataFrame, followup_df: pd.DataFrame) -> list[str]:
    keys = ["_health_signal_date", "_health_code"]
    for column in ["_health_capture_type", "_health_snapshot_time"]:
        signal_has_values = column in signal_df.columns and signal_df[column].fillna("").astype(str).ne("").any()
        followup_has_values = column in followup_df.columns and followup_df[column].fillna("").astype(str).ne("").any()
        if signal_has_values and followup_has_values:
            keys.append(column)
    return keys


def _health_snapshot(history_df: pd.DataFrame, current_date: str, cutoff_date: str | None = None) -> dict[str, object]:
    columns = {"tail_next_close_pct", "settled_tail_next_day", "_health_signal_date"}
    if history_df.empty or not columns.issubset(history_df.columns):
        return {
            "status": "样本少",
            "is_bad": False,
            "valid_count": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "loss5_rate_pct": None,
        }

    current_ts = pd.to_datetime(cutoff_date or current_date, errors="coerce")
    valid = history_df.copy()
    if pd.notna(current_ts):
        signal_ts = pd.to_datetime(valid["_health_signal_date"], errors="coerce")
        valid = valid[signal_ts.notna() & signal_ts.lt(current_ts)].copy()

    if valid.empty:
        return {
            "status": "样本少",
            "is_bad": False,
            "valid_count": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "loss5_rate_pct": None,
        }

    valid["tail_next_close_pct"] = pd.to_numeric(valid["tail_next_close_pct"], errors="coerce")
    valid = valid[_truthy_mask(valid["settled_tail_next_day"]) & valid["tail_next_close_pct"].notna()].copy()
    if valid.empty:
        return {
            "status": "样本少",
            "is_bad": False,
            "valid_count": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "loss5_rate_pct": None,
        }

    dates = sorted(date for date in valid["_health_signal_date"].dropna().astype(str).unique() if date)
    selected_dates = dates[-V3_HEALTH_WINDOW:]
    sample = valid[valid["_health_signal_date"].isin(selected_dates)].copy()
    returns = sample["tail_next_close_pct"].dropna()
    valid_count = int(len(returns))
    if valid_count <= 0:
        return {
            "status": "样本少",
            "is_bad": False,
            "valid_count": 0,
            "win_rate_pct": None,
            "avg_return_pct": None,
            "loss5_rate_pct": None,
        }

    win_rate_pct = round(float((returns > 0).mean() * 100), 2)
    avg_return_pct = round(float(returns.mean()), 2)
    loss5_rate_pct = round(float((returns <= -5).mean() * 100), 2)
    is_bad = (
        valid_count >= V3_HEALTH_MIN_VALID
        and (
            win_rate_pct < V3_HEALTH_WIN_RATE_FLOOR
            or avg_return_pct < V3_HEALTH_AVG_RETURN_FLOOR
            or loss5_rate_pct >= V3_HEALTH_LOSS5_CEILING
        )
    )
    status = "降权" if is_bad else ("样本少" if valid_count < V3_HEALTH_MIN_VALID else "正常")
    return {
        "status": status,
        "is_bad": is_bad,
        "valid_count": valid_count,
        "win_rate_pct": win_rate_pct,
        "avg_return_pct": avg_return_pct,
        "loss5_rate_pct": loss5_rate_pct,
    }


def apply_v3_health_cooldown(
    signal_df: pd.DataFrame,
    followup_df: pd.DataFrame,
    strategy_version: str = "v3",
) -> pd.DataFrame:
    if signal_df.empty or normalize_strategy_version(strategy_version) != "v3":
        return signal_df.copy()
    if followup_df.empty or not {"signal_date", "code"}.issubset(followup_df.columns):
        return signal_df.copy()

    result = _add_key_columns(signal_df)
    followups = _add_key_columns(followup_df)
    key_columns = _health_key_columns(result, followups)
    followup_lookup = followups.drop_duplicates(key_columns, keep="last").copy()

    for column, default in [
        ("v3_health_status", "样本少"),
        ("v3_health_cooldown", False),
        ("v3_health_valid_count", 0),
        ("v3_health_win_rate_pct", None),
        ("v3_health_avg_return_pct", None),
        ("v3_health_loss5_rate_pct", None),
    ]:
        if column not in result.columns:
            result[column] = default
    for column in ["risks", "suggested_action", "push_level", "is_pushed", "emotion_score"]:
        if column not in result.columns:
            result[column] = None

    selected_history: list[pd.DataFrame] = []
    signal_dates = sorted(date for date in result["_health_signal_date"].dropna().astype(str).unique() if date)
    for date_index, signal_date in enumerate(signal_dates):
        current_index = result.index[result["_health_signal_date"].astype(str).eq(signal_date)]
        if len(current_index) == 0:
            continue

        history_df = pd.concat(selected_history, ignore_index=True) if selected_history else pd.DataFrame()
        cutoff_date = signal_dates[date_index - 1] if date_index > 0 else signal_date
        health = _health_snapshot(history_df, signal_date, cutoff_date=cutoff_date)
        result.loc[current_index, "v3_health_status"] = health["status"]
        result.loc[current_index, "v3_health_valid_count"] = health["valid_count"]
        result.loc[current_index, "v3_health_win_rate_pct"] = health["win_rate_pct"]
        result.loc[current_index, "v3_health_avg_return_pct"] = health["avg_return_pct"]
        result.loc[current_index, "v3_health_loss5_rate_pct"] = health["loss5_rate_pct"]

        current = result.loc[current_index].copy()
        if bool(health["is_bad"]):
            scores = pd.to_numeric(current["emotion_score"], errors="coerce")
            pushed = _truthy_mask(current["is_pushed"])
            cooldown_index = current.index[pushed & scores.ge(V3_HEALTH_COOLDOWN_SCORE)]
            if len(cooldown_index) > 0:
                result.loc[cooldown_index, "is_pushed"] = False
                result.loc[cooldown_index, "push_level"] = OBSERVATION_POOL_LEVEL
                result.loc[cooldown_index, "v3_health_cooldown"] = True
                result.loc[cooldown_index, "risks"] = result.loc[cooldown_index, "risks"].apply(
                    lambda value: _append_note(value, V3_HEALTH_RISK_NOTE)
                )
                result.loc[cooldown_index, "suggested_action"] = result.loc[cooldown_index, "suggested_action"].apply(
                    lambda value: _append_note(value, V3_HEALTH_ACTION)
                )

        selected_current = result.loc[current_index].copy()
        selected_current = selected_current[_truthy_mask(selected_current["is_pushed"])].copy()
        if not selected_current.empty:
            selected_followups = selected_current[key_columns].merge(followup_lookup, on=key_columns, how="left")
            if not selected_followups.empty:
                selected_history.append(selected_followups)

    helper_columns = [column for column in result.columns if column.startswith("_health_")]
    return result.drop(columns=helper_columns, errors="ignore").reset_index(drop=True)
