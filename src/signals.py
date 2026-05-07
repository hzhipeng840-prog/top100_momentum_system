from __future__ import annotations

import pandas as pd

from src.market_regime import attach_market_regime
from src.paths import FEATURES_CSV, signals_csv_for
from src.strategy_profiles import DEFAULT_STRATEGY_VERSION, normalize_strategy_version, strategy_thresholds
from src.utils import parse_number, read_csv_safely, write_csv


OBSERVATION_POOL_LEVEL = "观察池"


def _add_reason(reasons: list[str], text: str) -> None:
    if text and text not in reasons:
        reasons.append(text)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _rank_score(rank: float | None) -> tuple[int, str | None]:
    if rank is None:
        return 0, None
    if rank <= 3:
        return 30, "人气Top3"
    if rank <= 10:
        return 24, "人气Top10"
    if rank <= 20:
        return 16, "人气Top20"
    if rank <= 50:
        return 8, "人气Top50"
    return 0, None


def _value_in_range(value: float | None, low: float | None = None, high: float | None = None) -> bool:
    if value is None:
        return False
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True


def _push_level_from_score(score: float, strategy_version: str, min_score: float) -> tuple[str, str]:
    watch_threshold, focus_threshold, strong_threshold = strategy_thresholds(strategy_version, min_score=min_score)
    tail_mode = strategy_version == "v3"
    winner_mode = strategy_version == "v4"

    if score >= strong_threshold:
        if tail_mode:
            return "强推观察", "尾盘版：只看临近收盘仍强承接的票，次日高开先看兑现，分歧后还能回封或稳住再留。"
        if winner_mode:
            return "强推观察", "强势归纳版：优先看早段强势且承接干净的票，重点盯 3 到 5 日延续，不追已经走到后段的人气股。"
        return "强推观察", "高风险情绪股：不追一字，优先看分歧承接，跌破5日线或人气退潮就跑。"
    if score >= focus_threshold:
        if tail_mode:
            return "重点观察", "尾盘版：优先留给收盘前强势股，次日高开不追，承接强再拿，承接弱先走。"
        if winner_mode:
            return "重点观察", "强势归纳版：先保留高胜率形态，等次日承接继续确认；一旦变成后段弱承接，就不再恋战。"
        return "重点观察", "只做观察或轻仓试错，等待回落承接和量能不失控。"
    if score >= watch_threshold:
        if tail_mode:
            return "普通观察", "尾盘版：保留在候选池，尽量只看尾盘收回高位或缩量稳住的票。"
        if winner_mode:
            return "普通观察", "强势归纳版：保留在候选池，优先跟踪早段强势和缩量承接，不主动接后段退潮样本。"
        return "普通观察", "保留在样本池，继续看次日人气和承接。"
    if tail_mode:
        return "不推送", "尾盘版：不主动追尾盘，等下一次收盘强承接或次日确认。"
    if winner_mode:
        return "不推送", "强势归纳版：这类形态历史胜率不占优，先留给更干净的强势样本。"
    return "不推送", "不主动推送。"


def _apply_v2_bonus_rules(row: pd.Series, reasons: list[str]) -> float:
    rank = parse_number(row.get("rank"))
    day_return = parse_number(row.get("day_return_pct"))
    close_position = parse_number(row.get("close_position"))
    volume_ratio = parse_number(row.get("volume_ratio_5"))
    pre5_return = parse_number(row.get("pre5_return_pct"))
    dist_ma20 = parse_number(row.get("dist_ma20_pct"))
    consecutive_days = parse_number(row.get("consecutive_days"))
    rank_change = parse_number(row.get("rank_change"))
    upper_shadow = parse_number(row.get("upper_shadow_pct"))
    limit_up_like = _as_bool(row.get("limit_up_like"))
    one_word_like = _as_bool(row.get("one_word_like"))

    bonus = 0.0

    continuation_signal = (
        rank is not None
        and rank <= 50
        and _value_in_range(day_return, 9.5, None)
        and _value_in_range(close_position, 0.9, None)
        and _value_in_range(pre5_return, 8, 35)
        and _value_in_range(volume_ratio, 0.55, 3.2)
        and _value_in_range(dist_ma20, 0, 35)
        and not one_word_like
    )
    if continuation_signal:
        bonus += 10
        _add_reason(reasons, "v2强势延续修正")

    hot_list_continuation = (
        rank is not None
        and rank <= 50
        and consecutive_days is not None
        and 2 <= consecutive_days <= 3
        and _value_in_range(pre5_return, 10, 40)
        and _value_in_range(volume_ratio, 0.7, 2.6)
        and _value_in_range(dist_ma20, 8, 35)
        and (rank_change is None or rank_change >= -10)
        and (day_return is None or day_return >= 0 or limit_up_like)
        and (close_position is None or close_position >= 0.55)
    )
    if hot_list_continuation:
        bonus += 8
        _add_reason(reasons, "v2连榜升温修正")

    hot_pullback_watch = (
        rank is not None
        and rank <= 100
        and consecutive_days is not None
        and 2 <= consecutive_days <= 4
        and _value_in_range(pre5_return, 10, 30)
        and _value_in_range(volume_ratio, 0.6, 2.2)
        and _value_in_range(dist_ma20, -5, 25)
        and _value_in_range(day_return, -4, 2)
        and _value_in_range(close_position, None, 0.35)
        and not one_word_like
        and (upper_shadow is None or upper_shadow <= 0.45)
    )
    if hot_pullback_watch:
        bonus += 14
        _add_reason(reasons, "v2热点回踩观察")

    if _value_in_range(day_return, 9.5, None) and limit_up_like and not one_word_like and _value_in_range(pre5_return, 12, 35):
        bonus += 6
        _add_reason(reasons, "v2低分涨停修正")

    return bonus


def _apply_v3_bonus_rules(row: pd.Series, reasons: list[str], risks: list[str]) -> float:
    rank = parse_number(row.get("rank"))
    day_return = parse_number(row.get("day_return_pct"))
    close_position = parse_number(row.get("close_position"))
    volume_ratio = parse_number(row.get("volume_ratio_5"))
    pre5_return = parse_number(row.get("pre5_return_pct"))
    dist_ma20 = parse_number(row.get("dist_ma20_pct"))
    consecutive_days = parse_number(row.get("consecutive_days"))
    upper_shadow = parse_number(row.get("upper_shadow_pct"))
    limit_up_like = _as_bool(row.get("limit_up_like"))
    one_word_like = _as_bool(row.get("one_word_like"))

    adjustment = 0.0

    tail_limit_continue = (
        rank is not None
        and rank <= 50
        and limit_up_like
        and not one_word_like
        and _value_in_range(close_position, 0.8, None)
        and _value_in_range(volume_ratio, 0.4, 1.1)
    )
    if tail_limit_continue:
        adjustment += 12
        _add_reason(reasons, "v3尾盘涨停延续")

    tail_low_volume_squeeze = (
        _value_in_range(close_position, 0.8, None)
        and _value_in_range(volume_ratio, 0.4, 1.0)
        and consecutive_days is not None
        and 1 <= consecutive_days <= 3
        and (upper_shadow is None or upper_shadow <= 0.35)
        and not one_word_like
    )
    if tail_low_volume_squeeze:
        adjustment += 10
        _add_reason(reasons, "v3尾盘缩量强承接")

    tail_hot_rank_early = (
        rank is not None
        and rank <= 50
        and _value_in_range(pre5_return, 5, 30)
        and _value_in_range(dist_ma20, 0, 28)
        and consecutive_days is not None
        and 1 <= consecutive_days <= 3
    )
    if tail_hot_rank_early:
        adjustment += 6
        _add_reason(reasons, "v3热点早段优先")

    if consecutive_days is not None and consecutive_days >= 5:
        adjustment -= 6
        _add_reason(risks, "v3尾盘版回避高位长连榜")

    if volume_ratio is not None and volume_ratio > 2.5:
        adjustment -= 6
        _add_reason(risks, "v3尾盘版回避放量过猛")

    if close_position is not None and 0.35 <= close_position <= 0.6 and day_return is not None and day_return > 0:
        adjustment -= 4
        _add_reason(risks, "v3尾盘承接一般")

    if day_return is not None and 7 <= day_return < 9.5 and (close_position is None or close_position < 0.8):
        adjustment -= 4
        _add_reason(risks, "v3尾盘半板但封单不强")

    return adjustment


def _apply_v4_bonus_rules(row: pd.Series, reasons: list[str], risks: list[str]) -> float:
    rank = parse_number(row.get("rank"))
    day_return = parse_number(row.get("day_return_pct"))
    close_position = parse_number(row.get("close_position"))
    volume_ratio = parse_number(row.get("volume_ratio_5"))
    pre5_return = parse_number(row.get("pre5_return_pct"))
    dist_ma20 = parse_number(row.get("dist_ma20_pct"))
    consecutive_days = parse_number(row.get("consecutive_days"))
    rank_change = parse_number(row.get("rank_change"))
    upper_shadow = parse_number(row.get("upper_shadow_pct"))
    one_word_like = _as_bool(row.get("one_word_like"))

    adjustment = 0.0

    winner_limit_follow_through = (
        _value_in_range(day_return, 9.5, None)
        and _value_in_range(close_position, 0.9, None)
        and _value_in_range(volume_ratio, 0.8, 1.8)
        and consecutive_days is not None
        and 2 <= consecutive_days <= 3
        and not one_word_like
    )
    if winner_limit_follow_through:
        adjustment += 14
        _add_reason(reasons, "v4强势连板胜率提纯")

    winner_early_range = (
        _value_in_range(day_return, 9.5, None)
        and _value_in_range(pre5_return, 10, 30)
        and _value_in_range(dist_ma20, 0, 28)
        and consecutive_days is not None
        and 1 <= consecutive_days <= 3
        and (rank_change is None or rank_change >= -5)
        and not one_word_like
    )
    if winner_early_range:
        adjustment += 12
        _add_reason(reasons, "v4早段强势区间命中")

    winner_squeeze_breakout = (
        _value_in_range(volume_ratio, 0.5, 1.2)
        and _value_in_range(pre5_return, 10, 30)
        and _value_in_range(dist_ma20, 0, 28)
        and _value_in_range(close_position, 0.75, None)
        and consecutive_days is not None
        and 2 <= consecutive_days <= 3
        and (upper_shadow is None or upper_shadow <= 0.3)
    )
    if winner_squeeze_breakout:
        adjustment += 10
        _add_reason(reasons, "v4缩量突破胜率区间")

    weak_late_board = (
        rank is not None
        and rank > 50
        and consecutive_days is not None
        and consecutive_days >= 4
        and _value_in_range(close_position, 0.45, 0.75)
    )
    if weak_late_board:
        adjustment -= 16
        _add_reason(risks, "v4回避后段弱承接长连榜")

    weak_cold_pullback = (
        rank is not None
        and rank > 50
        and _value_in_range(day_return, -4, 2)
        and pre5_return is not None
        and pre5_return < 0
    )
    if weak_cold_pullback:
        adjustment -= 10
        _add_reason(risks, "v4回避冷启动弱回踩")

    weak_mid_close = (
        day_return is not None
        and day_return < 0
        and _value_in_range(close_position, 0.45, 0.75)
    )
    if weak_mid_close:
        adjustment -= 10
        _add_reason(risks, "v4回避阴线中段承接")

    overheated_late = (
        pre5_return is not None
        and pre5_return > 30
        and consecutive_days is not None
        and consecutive_days >= 4
    )
    if overheated_late:
        adjustment -= 8
        _add_reason(risks, "v4回避高位后段透支")

    late_rank_fade = (
        rank is not None
        and rank > 50
        and consecutive_days is not None
        and consecutive_days >= 4
        and rank_change is not None
        and rank_change <= -10
    )
    if late_rank_fade:
        adjustment -= 8
        _add_reason(risks, "v4回避人气退潮后段样本")

    return adjustment


def _market_regime_adjustment(row: pd.Series, reasons: list[str], risks: list[str], strategy_version: str) -> float:
    market_regime = str(row.get("market_regime") or "").strip()
    if market_regime in {"", "未知"}:
        return 0.0

    relative_1d = parse_number(row.get("relative_1d_pct"))
    relative_5d = parse_number(row.get("relative_5d_pct"))
    market_1d = parse_number(row.get("market_1d_pct"))
    market_5d = parse_number(row.get("market_5d_pct"))
    day_return = parse_number(row.get("day_return_pct"))
    close_position = parse_number(row.get("close_position"))
    rank = parse_number(row.get("rank"))
    consecutive_days = parse_number(row.get("consecutive_days"))
    one_word_like = _as_bool(row.get("one_word_like"))

    version_weight = {
        "v1": 0.75,
        "v2": 1.0,
        "v3": 1.15,
        "v4": 1.1,
    }.get(strategy_version, 1.0)

    adjustment = 0.0
    relative_strength = False
    relative_weakness = False

    if relative_1d is not None or relative_5d is not None:
        relative_strength = (
            (relative_1d is not None and relative_1d >= 1.0)
            or (relative_5d is not None and relative_5d >= 2.0)
        )
        relative_weakness = (
            (relative_1d is not None and relative_1d <= -1.0)
            or (relative_5d is not None and relative_5d <= -2.0)
        )

    if market_regime == "强势":
        if relative_strength:
            adjustment += 4.0 * version_weight
            _add_reason(reasons, "市场强势且个股相对强势")
        elif relative_weakness:
            adjustment -= 3.0 * version_weight
            _add_reason(risks, "强势市里个股相对走弱")
    elif market_regime == "震荡":
        if relative_strength:
            adjustment += 3.0 * version_weight
            _add_reason(reasons, "震荡市里保持相对强势")
        elif relative_weakness:
            adjustment -= 6.0 * version_weight
            _add_reason(risks, "震荡市里相对走弱")

        if (
            day_return is not None
            and day_return > 0
            and close_position is not None
            and close_position >= 0.75
            and (market_1d is None or market_1d <= 1.0)
            and (market_5d is None or market_5d <= 1.5)
        ):
            adjustment += 1.5 * version_weight
            _add_reason(reasons, "震荡市里承接未失控")
    elif market_regime == "弱势":
        if relative_strength and close_position is not None and close_position >= 0.75 and not one_word_like:
            adjustment += 2.0 * version_weight
            _add_reason(reasons, "弱市中个股相对抗跌")
        else:
            adjustment -= 8.0 * version_weight
            _add_reason(risks, "弱市先收缩执行")

    if rank is not None and rank > 50 and market_regime in {"震荡", "弱势"}:
        adjustment -= 1.5 * version_weight
        _add_reason(risks, "后段人气股在偏弱环境里更谨慎")

    if consecutive_days is not None and consecutive_days >= 5 and market_regime == "弱势":
        adjustment -= 2.0 * version_weight
        _add_reason(risks, "弱市里的长连榜样本优先降权")

    return adjustment


def score_signal(
    row: pd.Series,
    min_score: float = 60,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> dict:
    strategy_version = normalize_strategy_version(strategy_version)
    score = 0.0
    reasons: list[str] = []
    risks: list[str] = []

    rank = parse_number(row.get("rank"))
    points, rank_reason = _rank_score(rank)
    score += points
    if rank_reason:
        _add_reason(reasons, rank_reason)

    day_return = parse_number(row.get("day_return_pct"))
    if day_return is not None:
        if day_return >= 9.5:
            score += 22
            _add_reason(reasons, "当日涨停附近")
        elif day_return >= 7:
            score += 16
            _add_reason(reasons, "当日强涨")
        elif day_return >= 3:
            score += 8
            _add_reason(reasons, "当日上涨")
        elif day_return < 0:
            score -= 10
            _add_reason(risks, "当日转弱")

    close_position = parse_number(row.get("close_position"))
    if close_position is not None:
        if close_position >= 0.9:
            score += 16
            _add_reason(reasons, "收在日内高位")
        elif close_position >= 0.75:
            score += 10
            _add_reason(reasons, "收盘承接较强")
        elif close_position < 0.45:
            score -= 12
            _add_reason(risks, "收盘承接偏弱")

    volume_ratio = parse_number(row.get("volume_ratio_5"))
    if volume_ratio is not None:
        if 0.8 <= volume_ratio <= 1.8:
            score += 14
            _add_reason(reasons, "温和放量")
        elif 1.8 < volume_ratio <= 2.6:
            score += 8
            _add_reason(reasons, "放量仍可控")
        elif volume_ratio > 3.5:
            score -= 12
            _add_reason(risks, "放量过猛")
        elif volume_ratio < 0.5:
            score -= 8
            _add_reason(risks, "量能不足")

    pre5_return = parse_number(row.get("pre5_return_pct"))
    if pre5_return is not None:
        if 5 <= pre5_return <= 25:
            score += 12
            _add_reason(reasons, "近5日趋势升温")
        elif 25 < pre5_return <= 45:
            score += 5
            _add_reason(reasons, "近5日强势延续")
            _add_reason(risks, "短线涨幅偏高")
        elif pre5_return > 45:
            score -= 8
            _add_reason(risks, "短线涨幅过热")
        elif pre5_return < 0:
            score -= 4
            _add_reason(risks, "近5日趋势未确认")

    dist_ma20 = parse_number(row.get("dist_ma20_pct"))
    if dist_ma20 is not None:
        if 0 <= dist_ma20 <= 28:
            score += 8
            _add_reason(reasons, "站上20日线且不过热")
        elif 28 < dist_ma20 <= 45:
            score -= 2
            _add_reason(risks, "偏离20日线较大")
        elif dist_ma20 > 45:
            score -= 12
            _add_reason(risks, "远离20日线过多")
        elif dist_ma20 < -5:
            score -= 8
            _add_reason(risks, "仍在20日线下方")

    consecutive_days = parse_number(row.get("consecutive_days"))
    if consecutive_days is not None:
        if 2 <= consecutive_days <= 4:
            score += 8
            _add_reason(reasons, "连续上榜确认")
        elif consecutive_days >= 5:
            score += 4
            _add_reason(reasons, "持续霸榜")
            _add_reason(risks, "热度可能进入后段")

    rank_change = parse_number(row.get("rank_change"))
    if rank_change is not None:
        if rank_change >= 10:
            score += 4
            _add_reason(reasons, "人气排名明显上升")
        elif rank_change <= -20:
            score -= 4
            _add_reason(risks, "人气排名明显回落")

    one_word_like = _as_bool(row.get("one_word_like"))
    if one_word_like:
        score -= 6
        _add_reason(risks, "一字或近似一字，参与点差")

    upper_shadow = parse_number(row.get("upper_shadow_pct"))
    if upper_shadow is not None and upper_shadow >= 0.45 and day_return is not None and day_return > 0:
        score -= 6
        _add_reason(risks, "上影线偏长")

    price_status = str(row.get("price_status") or "")
    if price_status != "ok":
        score -= 20
        _add_reason(risks, "缺少当日行情")

    appearance_count = parse_number(row.get("appearance_count"))
    if appearance_count == 1 and rank is not None and rank <= 20 and day_return is not None and day_return >= 7:
        score += 6
        _add_reason(reasons, "首次上榜即强势")
        if (
            strategy_version in {"v2", "v3"}
            and close_position is not None
            and close_position >= 0.75
            and (volume_ratio is None or 0.5 <= volume_ratio <= 2.5)
            and not one_word_like
        ):
            score += 8
            _add_reason(reasons, "首次上榜强势提权")

    score += _market_regime_adjustment(row, reasons=reasons, risks=risks, strategy_version=strategy_version)

    if strategy_version in {"v1", "v2", "v3", "v4"}:
        score += _apply_v2_bonus_rules(row, reasons=reasons)
    if strategy_version == "v3":
        score += _apply_v3_bonus_rules(row, reasons=reasons, risks=risks)
        score += _apply_v4_bonus_rules(row, reasons=reasons, risks=risks)
    if strategy_version == "v4":
        score += _apply_v4_bonus_rules(row, reasons=reasons, risks=risks)

    score = round(max(score, 0), 2)
    push_level, action = _push_level_from_score(score, strategy_version=strategy_version, min_score=min_score)
    watch_threshold, _, _ = strategy_thresholds(strategy_version, min_score=min_score)
    observation_pool = one_word_like and price_status == "ok"
    if observation_pool:
        push_level = OBSERVATION_POOL_LEVEL
        action = "封板不可买，放观察池跟踪"
    return {
        "emotion_score": score,
        "push_level": push_level,
        "is_pushed": bool(score >= watch_threshold and price_status == "ok" and not observation_pool),
        "reasons": "；".join(reasons) if reasons else "-",
        "risks": "；".join(risks) if risks else "-",
        "suggested_action": action,
        "strategy_version": strategy_version,
    }


def build_signals(
    feature_df: pd.DataFrame | None = None,
    min_score: float = 60,
    market_regime_df: pd.DataFrame | None = None,
    strategy_version: str = DEFAULT_STRATEGY_VERSION,
) -> pd.DataFrame:
    strategy_version = normalize_strategy_version(strategy_version)
    df = read_csv_safely(FEATURES_CSV) if feature_df is None else feature_df.copy()
    if df.empty:
        return pd.DataFrame()

    df = attach_market_regime(df, market_regime_df=market_regime_df)
    scored = pd.DataFrame([score_signal(row, min_score=min_score, strategy_version=strategy_version) for _, row in df.iterrows()])
    result = pd.concat([df.reset_index(drop=True), scored], axis=1)
    result["strategy_version"] = strategy_version
    return result.sort_values(["signal_date", "emotion_score", "rank"], ascending=[True, False, True]).reset_index(drop=True)


def save_signals(signal_df: pd.DataFrame, strategy_version: str = DEFAULT_STRATEGY_VERSION) -> None:
    write_csv(signal_df, signals_csv_for(strategy_version))
