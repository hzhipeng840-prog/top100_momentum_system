from __future__ import annotations

import unittest

from src.paths import followups_csv_for, signals_csv_for
from src.signals import score_signal
from src.strategy_profiles import (
    strategy_capture_priority,
    strategy_default_metric_label,
    strategy_thresholds,
)


class StrategyVersioningTest(unittest.TestCase):
    def test_versioned_paths_keep_v1_and_split_v2_v3(self) -> None:
        self.assertTrue(str(signals_csv_for("v1")).endswith("signals.csv"))
        self.assertTrue(str(signals_csv_for("v2")).endswith("signals_v2.csv"))
        self.assertTrue(str(signals_csv_for("v3")).endswith("signals_v3.csv"))
        self.assertTrue(str(followups_csv_for("v2")).endswith("followups_v2.csv"))
        self.assertTrue(str(followups_csv_for("v3")).endswith("followups_v3.csv"))

    def test_v2_can_promote_continuation_candidate(self) -> None:
        row = {
            "rank": 37,
            "day_return_pct": 10.01,
            "close_position": 1.0,
            "volume_ratio_5": 0.65,
            "pre5_return_pct": 24.0,
            "dist_ma20_pct": 15.0,
            "consecutive_days": 1,
            "appearance_count": 1,
            "rank_change": 6,
            "one_word_like": False,
            "limit_up_like": True,
            "upper_shadow_pct": 0.02,
            "price_status": "ok",
        }

        v1 = score_signal(row, strategy_version="v1")
        v2 = score_signal(row, strategy_version="v2")

        self.assertGreater(v2["emotion_score"], v1["emotion_score"])
        self.assertNotEqual(v2["push_level"], v1["push_level"])
        self.assertIn("v2", "".join(v2["reasons"]))

    def test_v3_raises_tail_candidate_and_uses_tail_defaults(self) -> None:
        row = {
            "rank": 28,
            "day_return_pct": 10.02,
            "close_position": 0.96,
            "volume_ratio_5": 0.82,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 12.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 4,
            "one_word_like": False,
            "limit_up_like": True,
            "upper_shadow_pct": 0.1,
            "price_status": "ok",
        }

        v2 = score_signal(row, strategy_version="v2")
        v3 = score_signal(row, strategy_version="v3")

        self.assertGreater(v3["emotion_score"], v2["emotion_score"])
        self.assertTrue(v3["is_pushed"])
        self.assertIn("v3", "".join(v3["reasons"]))
        self.assertEqual(strategy_default_metric_label("v3"), "次日收盘收益")
        self.assertEqual(strategy_thresholds("v3", min_score=60), (65.0, 78.0, 90.0))
        self.assertEqual(strategy_capture_priority("v3"), ["intraday_1430", "post_close"])


if __name__ == "__main__":
    unittest.main()
