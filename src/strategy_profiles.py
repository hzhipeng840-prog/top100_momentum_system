from __future__ import annotations

from copy import deepcopy


DEFAULT_STRATEGY_VERSION = "v1"


STRATEGY_PROFILES: dict[str, dict[str, object]] = {
    "v1": {
        "version": "v1",
        "name": "v1 当前策略",
        "description": "当前收盘情绪动量规则，偏重当日强度与收盘承接。",
        "watch_threshold": 60.0,
        "focus_threshold": 75.0,
        "strong_threshold": 90.0,
        "default_metric_label": "5日收益",
        "capture_priority": ["post_close"],
    },
    "v2": {
        "version": "v2",
        "name": "v2 强势修正版",
        "description": "参考强势复盘里的漏选样本，额外识别强势延续、连榜升温和热点回踩观察。",
        "watch_threshold": 60.0,
        "focus_threshold": 75.0,
        "strong_threshold": 90.0,
        "default_metric_label": "5日收益",
        "capture_priority": ["post_close"],
    },
    "v3": {
        "version": "v3",
        "name": "v3 尾盘版",
        "description": "偏重尾盘承接与次日延续，主要评估尾盘买入后的次日开盘/收盘表现。",
        "watch_threshold": 65.0,
        "focus_threshold": 78.0,
        "strong_threshold": 90.0,
        "default_metric_label": "次日收盘收益",
        "capture_priority": ["intraday_1430", "post_close"],
    },
}


def normalize_strategy_version(version: object) -> str:
    text = str(version or DEFAULT_STRATEGY_VERSION).strip().lower()
    if not text:
        return DEFAULT_STRATEGY_VERSION
    return text if text in STRATEGY_PROFILES else DEFAULT_STRATEGY_VERSION


def get_strategy_profile(version: object) -> dict[str, object]:
    normalized = normalize_strategy_version(version)
    return deepcopy(STRATEGY_PROFILES[normalized])


def available_strategy_versions(settings: dict | None = None) -> list[str]:
    configured = settings.get("strategy_versions") if isinstance(settings, dict) else None
    if not isinstance(configured, list) or not configured:
        return list(STRATEGY_PROFILES.keys())

    versions: list[str] = []
    for version in configured:
        normalized = normalize_strategy_version(version)
        if normalized not in versions:
            versions.append(normalized)
    if DEFAULT_STRATEGY_VERSION not in versions:
        versions.insert(0, DEFAULT_STRATEGY_VERSION)
    return versions


def strategy_thresholds(version: object, min_score: float | int = 60) -> tuple[float, float, float]:
    profile = get_strategy_profile(version)
    watch_threshold = max(float(profile.get("watch_threshold", 60.0)), float(min_score))
    focus_threshold = max(float(profile.get("focus_threshold", 75.0)), watch_threshold)
    strong_threshold = max(float(profile.get("strong_threshold", 90.0)), focus_threshold)
    return watch_threshold, focus_threshold, strong_threshold


def strategy_default_metric_label(version: object) -> str:
    profile = get_strategy_profile(version)
    return str(profile.get("default_metric_label", "5日收益"))


def strategy_capture_priority(version: object) -> list[str]:
    profile = get_strategy_profile(version)
    values = profile.get("capture_priority", ["post_close"])
    if not isinstance(values, list) or not values:
        return ["post_close"]
    return [str(value).strip() for value in values if str(value).strip()]
