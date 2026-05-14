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


def _buyability_gate(row: pd.Series) -> tuple[bool, str | None]:
    one_word_like = _as_bool(row.get("one_word_like"))
    if one_word_like:
        return False, "一字板不可买"

    limit_up_like = _as_bool(row.get("limit_up_like"))
    if limit_up_like:
        return False, "涨停封板不可买"

    day_return = parse_number(row.get("day_return_pct"))
    close_position = parse_number(row.get("close_position"))
    if day_return is not None and day_return >= 9.5 and (close_position is None or close_position >= 0.9):
        return False, "涨停封板不可买"

    return True, None


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


def _apply_v4_technical_rules(row: pd.Series, reasons: list[str], risks: list[str]) -> float:
    day_return = parse_number(row.get("day_return_pct"))
    pre3_return = parse_number(row.get("pre3_return_pct"))
    pre5_return = parse_number(row.get("pre5_return_pct"))
    close_position = parse_number(row.get("close_position"))
    volume_ratio = parse_number(row.get("volume_ratio_5"))
    dist_ma5 = parse_number(row.get("dist_ma5_pct"))
    dist_ma10 = parse_number(row.get("dist_ma10_pct"))
    dist_ma20 = parse_number(row.get("dist_ma20_pct"))
    upper_shadow = parse_number(row.get("upper_shadow_pct"))
    rank = parse_number(row.get("rank"))
    consecutive_days = parse_number(row.get("consecutive_days"))
    limit_up_like = _as_bool(row.get("limit_up_like"))

    adjustment = 0.0

    bullish_alignment = (
        dist_ma5 is not None
        and dist_ma10 is not None
        and dist_ma20 is not None
        and dist_ma5 > 0
        and dist_ma10 > 0
        and dist_ma20 > 0
        and dist_ma5 < dist_ma10 < dist_ma20
    )
    if bullish_alignment:
        adjustment += 12.0
        _add_reason(reasons, "v4多头排列技术加分")

    ma5_pullback = (
        bullish_alignment
        and _value_in_range(dist_ma5, 0.0, 2.2)
        and _value_in_range(volume_ratio, 0.55, 1.15)
        and _value_in_range(close_position, 0.72, None)
        and _value_in_range(day_return, -1.5, 4.5)
    )
    if ma5_pullback:
        adjustment += 10.0
        _add_reason(reasons, "v4缩量回踩MA5")

    ma10_support = (
        bullish_alignment
        and _value_in_range(dist_ma10, 0.0, 4.2)
        and (dist_ma5 is None or dist_ma5 <= 2.8)
        and _value_in_range(volume_ratio, 0.55, 1.25)
        and _value_in_range(close_position, 0.7, None)
        and _value_in_range(day_return, -2.0, 4.5)
    )
    if ma10_support:
        adjustment += 8.0
        _add_reason(reasons, "v4靠近MA10承接")

    healthy_breakout = (
        bullish_alignment
        and not limit_up_like
        and _value_in_range(day_return, 4.0, 8.8)
        and _value_in_range(close_position, 0.82, None)
        and _value_in_range(volume_ratio, 0.85, 1.8)
        and _value_in_range(dist_ma5, 1.0, 4.8)
        and _value_in_range(pre5_return, 4.0, 18.0)
    )
    if healthy_breakout:
        adjustment += 6.0
        _add_reason(reasons, "v4放量突破但不过热")

    if bullish_alignment and _value_in_range(pre3_return, -2.0, 8.0) and _value_in_range(pre5_return, 3.0, 18.0):
        adjustment += 4.0
        _add_reason(reasons, "v4沿短均线抬升")

    if dist_ma5 is not None and dist_ma5 > 5.0:
        adjustment -= 12.0
        _add_reason(risks, "v4乖离率过大")
    elif dist_ma5 is not None and dist_ma5 > 3.0:
        adjustment -= 5.0
        _add_reason(risks, "v4短线偏离MA5过多")

    if (dist_ma10 is not None and dist_ma10 > 8.0) or (dist_ma20 is not None and dist_ma20 > 25.0):
        adjustment -= 8.0
        _add_reason(risks, "v4远离短中期均线")

    if volume_ratio is not None and volume_ratio > 2.0 and day_return is not None and day_return >= 7.0:
        adjustment -= 8.0
        _add_reason(risks, "v4放量冲高过热")

    if upper_shadow is not None and upper_shadow > 0.35 and (close_position is None or close_position < 0.82):
        adjustment -= 6.0
        _add_reason(risks, "v4长上影承接转弱")

    if close_position is not None and close_position < 0.58 and day_return is not None and day_return > 0:
        adjustment -= 8.0
        _add_reason(risks, "v4正涨但收盘承接一般")

    if (
        dist_ma5 is not None
        and dist_ma10 is not None
        and dist_ma5 < 0
        and dist_ma10 < 0
        and day_return is not None
        and day_return < 0
    ):
        adjustment -= 8.0
        _add_reason(risks, "v4跌破短均线")

    if (
        not bullish_alignment
        and rank is not None
        and rank > 20
        and consecutive_days is not None
        and consecutive_days >= 3
    ):
        adjustment -= 4.0
        _add_reason(risks, "v4趋势未转强且热度偏后")

    return adjustment


def _apply_v4_optional_flow_chip_rules(row: pd.Series, reasons: list[str], risks: list[str]) -> float:
    adjustment = 0.0

    concentration_90 = parse_number(row.get("concentration_90"))
    profit_ratio = parse_number(row.get("profit_ratio"))
    capital_flow_signal = parse_number(row.get("capital_flow_signal"))
    board_strength = parse_number(row.get("board_strength"))
    dragon_tiger_positive = _as_bool(row.get("dragon_tiger_positive"))
    limit_up_like = _as_bool(row.get("limit_up_like"))
    close_position = parse_number(row.get("close_position"))
    day_return = parse_number(row.get("day_return_pct"))

    if concentration_90 is not None and concentration_90 < 15:
        adjustment += 6.0
        _add_reason(reasons, "v4筹码集中加分")
    elif concentration_90 is not None and concentration_90 > 28:
        adjustment -= 6.0
        _add_reason(risks, "v4筹码分散")

    if profit_ratio is not None and 45 <= profit_ratio <= 82:
        adjustment += 4.0
        _add_reason(reasons, "v4获利盘结构健康")
    elif profit_ratio is not None and profit_ratio > 92:
        adjustment -= 6.0
        _add_reason(risks, "v4获利盘过热")
    elif profit_ratio is not None and profit_ratio < 22:
        adjustment -= 3.0
        _add_reason(risks, "v4获利盘结构偏弱")

    if capital_flow_signal is not None and capital_flow_signal >= 3.0:
        adjustment += 8.0
        _add_reason(reasons, "v4主力资金强净流入")
    elif capital_flow_signal is not None and capital_flow_signal > 0:
        adjustment += 4.0
        _add_reason(reasons, "v4主力资金净流入")
    elif capital_flow_signal is not None and capital_flow_signal <= -3.0:
        adjustment -= 8.0
        _add_reason(risks, "v4主力资金强净流出")
    elif capital_flow_signal is not None and capital_flow_signal < 0:
        adjustment -= 4.0
        _add_reason(risks, "v4主力资金净流出")

    if board_strength is not None and board_strength > 0:
        adjustment += 3.0
        _add_reason(reasons, "v4板块联动加分")

    if dragon_tiger_positive:
        if not limit_up_like and (close_position is None or close_position >= 0.7):
            adjustment += 5.0
            _add_reason(reasons, "v4龙虎榜活跃")
        else:
            adjustment += 2.0
            _add_reason(reasons, "v4龙虎榜提示关注")

    if (
        concentration_90 is not None
        and concentration_90 < 16
        and profit_ratio is not None
        and 45 <= profit_ratio <= 82
        and capital_flow_signal is not None
        and capital_flow_signal > 0
    ):
        adjustment += 5.0
        _add_reason(reasons, "v4筹码资金共振")

    if (
        capital_flow_signal is not None
        and capital_flow_signal < 0
        and day_return is not None
        and day_return > 0
        and (close_position is None or close_position < 0.75)
    ):
        adjustment -= 4.0
        _add_reason(risks, "v4拉升但资金未跟随")

    return adjustment


def _apply_v4_event_penalty(row: pd.Series, reasons: list[str], risks: list[str]) -> float:
    keyword_fields = [
        "announcement_summary",
        "event_summary",
        "news_summary",
        "company_highlights",
        "fundamental_notes",
        "risk_warning",
    ]
    negative_keywords = ("减持", "监管", "问询", "处罚", "立案", "警示函")
    caution_keywords = ("风险提示", "异常波动")
    positive_keywords = ("重大合同", "中标", "签订协议", "战略合作", "回购", "增持", "业绩预增")

    negative_hits: list[str] = []
    caution_hits: list[str] = []
    positive_hits: list[str] = []
    for field in keyword_fields:
        text = str(row.get(field) or "").strip()
        if not text:
            continue
        for keyword in negative_keywords:
            if keyword in text and keyword not in negative_hits:
                negative_hits.append(keyword)
        for keyword in caution_keywords:
            if keyword in text and keyword not in caution_hits:
                caution_hits.append(keyword)
        for keyword in positive_keywords:
            if keyword in text and keyword not in positive_hits:
                positive_hits.append(keyword)

    adjustment = 0.0
    if positive_hits:
        adjustment += min(5.0, 2.0 + 1.0 * len(positive_hits))
        _add_reason(reasons, f"v4事件催化加分（{'/'.join(positive_hits)}）")

    if caution_hits:
        caution_penalty = min(4.0, 2.0 + 1.0 * len(caution_hits))
        adjustment -= caution_penalty
        _add_reason(risks, f"v4事件谨慎扣分（{'/'.join(caution_hits)}）")

    if negative_hits:
        penalty = min(10.0, 4.0 + 2.0 * len(negative_hits))
        adjustment -= penalty
        _add_reason(risks, f"v4事件风险扣分（{'/'.join(negative_hits)}）")

    return adjustment


def _apply_v2_recap_executable_rules(
    row: pd.Series,
    reasons: list[str],
    risks: list[str],
) -> float:
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
    limit_up_like = _as_bool(row.get("limit_up_like"))

    base_bonus = 10.0
    follow_bonus = 7.0
    first_bonus = 4.0
    weak_penalty = 9.0
    late_penalty = 8.0
    fade_penalty = 6.0

    adjustment = 0.0

    early_warming_shape = (
        rank is not None
        and rank <= 30
        and _value_in_range(day_return, 3.0, 8.8)
        and _value_in_range(close_position, 0.75, 0.95)
        and _value_in_range(volume_ratio, 0.8, 1.6)
        and _value_in_range(pre5_return, 5.0, 18.0)
        and _value_in_range(dist_ma20, 0.0, 25.0)
        and not one_word_like
        and not limit_up_like
    )
    if early_warming_shape:
        adjustment += base_bonus
        _add_reason(reasons, "v2早段升温样本提权")

    follow_through_shape = (
        rank is not None
        and rank <= 20
        and consecutive_days is not None
        and 2 <= consecutive_days <= 4
        and _value_in_range(close_position, 0.75, 0.95)
        and (rank_change is None or (-5 <= rank_change <= 60))
        and (upper_shadow is None or upper_shadow <= 0.22)
        and _value_in_range(volume_ratio, 0.8, 1.6)
        and _value_in_range(pre5_return, 5.0, 22.0)
        and not one_word_like
        and not limit_up_like
    )
    if follow_through_shape:
        adjustment += follow_bonus
        _add_reason(reasons, "v2连榜确认样本提权")

    first_appearance_shape = (
        rank is not None
        and rank <= 15
        and _value_in_range(day_return, 3.0, 8.5)
        and _value_in_range(close_position, 0.75, 0.92)
        and _value_in_range(volume_ratio, 0.8, 1.4)
        and _value_in_range(pre5_return, 5.0, 16.0)
        and _value_in_range(dist_ma20, 0.0, 22.0)
        and not one_word_like
        and not limit_up_like
        and consecutive_days is not None
        and consecutive_days <= 2
    )
    if first_appearance_shape:
        adjustment += first_bonus
        _add_reason(reasons, "v2首次强势样本提权")

    weak_close_shape = (
        day_return is not None
        and day_return < 0
        and close_position is not None
        and close_position < 0.62
    )
    if weak_close_shape:
        adjustment -= weak_penalty
        _add_reason(risks, "v2回避转弱弱收盘样本")

    late_overheated_shape = (
        rank is not None
        and rank > 30
        and consecutive_days is not None
        and consecutive_days >= 4
        and pre5_return is not None
        and pre5_return > 20
    )
    if late_overheated_shape:
        adjustment -= late_penalty
        _add_reason(risks, "v2回避后段过热样本")

    rank_fade_shape = (
        rank_change is not None
        and rank_change <= -10
        and close_position is not None
        and close_position < 0.7
    )
    if rank_fade_shape:
        adjustment -= fade_penalty
        _add_reason(risks, "v2回避人气转弱样本")

    tight_nonlimit_shape = (
        not limit_up_like
        and _value_in_range(close_position, 0.95, None)
        and _value_in_range(day_return, 7.0, None)
    )
    if tight_nonlimit_shape:
        adjustment -= 8.0
        _add_reason(risks, "v2回避非封板高位追涨样本")

    stretched_buyable_shape = (
        not limit_up_like
        and rank is not None
        and rank > 20
        and volume_ratio is not None
        and volume_ratio > 1.5
    )
    if stretched_buyable_shape:
        adjustment -= 5.0
        _add_reason(risks, "v2回避中后段放量追高样本")

    overheated_first_shape = (
        not limit_up_like
        and consecutive_days is not None
        and consecutive_days <= 2
        and pre5_return is not None
        and pre5_return > 25.0
    )
    if overheated_first_shape:
        adjustment -= 6.0
        _add_reason(risks, "v2回避前段过热样本")

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

    if strategy_version in {"v1", "v2", "v3"}:
        score += _apply_v2_bonus_rules(row, reasons=reasons)
    if strategy_version == "v3":
        score += _apply_v3_bonus_rules(row, reasons=reasons, risks=risks)
    if strategy_version == "v4":
        score += _apply_v4_technical_rules(row, reasons=reasons, risks=risks)
        score += _apply_v4_optional_flow_chip_rules(row, reasons=reasons, risks=risks)
        score += _apply_v4_event_penalty(row, reasons=reasons, risks=risks)
    if strategy_version == "v2":
        score += _apply_v2_recap_executable_rules(
            row,
            reasons=reasons,
            risks=risks,
        )

    score = round(max(score, 0), 2)
    push_level, action = _push_level_from_score(score, strategy_version=strategy_version, min_score=min_score)
    watch_threshold, _, _ = strategy_thresholds(strategy_version, min_score=min_score)
    buyable, buyability_reason = _buyability_gate(row)
    if not buyable:
        _add_reason(risks, buyability_reason or "封板不可买")
    observation_pool = price_status == "ok" and not buyable
    if observation_pool:
        push_level = OBSERVATION_POOL_LEVEL
        action = f"{buyability_reason or '封板不可买'}，放观察池跟踪"
    return {
        "emotion_score": score,
        "push_level": push_level,
        "is_pushed": bool(score >= watch_threshold and price_status == "ok" and buyable),
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
