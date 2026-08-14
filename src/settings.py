from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.paths import CONFIG_PATH


DEFAULT_SETTINGS: dict[str, Any] = {
    "top_n": 100,
    "default_capture_type": "post_close",
    "default_strategy_version": "v1",
    "strategy_versions": ["v1", "v2", "v3", "v4"],
    "visible_strategy_versions": ["v1", "v2", "v3"],
    "signal_min_score": 60,
    "latest_push_limit": None,
    "strong_return_threshold_pct": 15,
    "followup_days": [1, 3, 5, 10],
    "settlement_freshness_min_ratio": 0.95,
    "settlement_backfill_batch_size": 80,
    "full_report_mode": "light",
    "enable_v4_context_in_full": False,
    "refresh_price_cache": True,
    "refresh_market_cache": True,
    "refresh_intraday_cache": False,
    "intraday_cache_push_only": True,
    "intraday_cache_limit": 20,
}


def load_settings(path: Path = CONFIG_PATH) -> dict[str, Any]:
    settings = DEFAULT_SETTINGS.copy()
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            loaded = {}
        if isinstance(loaded, dict):
            settings.update(loaded)
    return settings
