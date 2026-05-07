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
        self.assertTrue(str(signals_csv_for("v4")).endswith("signals_v4.csv"))
        self.assertTrue(str(followups_csv_for("v2")).endswith("followups_v2.csv"))
        self.assertTrue(str(followups_csv_for("v3")).endswith("followups_v3.csv"))
        self.assertTrue(str(followups_csv_for("v4")).endswith("followups_v4.csv"))

    def test_v1_absorbs_v2_bonus_rules(self) -> None:
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

        self.assertEqual(v1["emotion_score"], v2["emotion_score"])
        self.assertEqual(v1["push_level"], v2["push_level"])
        self.assertIn("v2", "".join(v1["reasons"]))
        self.assertIn("v2", "".join(v2["reasons"]))
        self.assertEqual(strategy_thresholds("v1", min_score=60), (62.0, 77.0, 92.0))

    def test_v2_rewards_first_appearance_more_than_repeated_strong_row(self) -> None:
        first_row = {
            "rank": 8,
            "day_return_pct": 10.02,
            "close_position": 0.96,
            "volume_ratio_5": 1.05,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 15.0,
            "consecutive_days": 1,
            "appearance_count": 1,
            "rank_change": 4,
            "one_word_like": False,
            "limit_up_like": True,
            "upper_shadow_pct": 0.08,
            "price_status": "ok",
        }
        repeat_row = dict(first_row, appearance_count=2)

        first = score_signal(first_row, strategy_version="v2")
        repeat = score_signal(repeat_row, strategy_version="v2")

        self.assertGreater(first["emotion_score"], repeat["emotion_score"])
        self.assertIn("首次上榜强势提权", first["reasons"])

    def test_one_word_like_moves_to_observation_pool(self) -> None:
        row = {
            "rank": 5,
            "day_return_pct": 10.02,
            "close_position": 0.98,
            "volume_ratio_5": 1.0,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 16.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 3,
            "one_word_like": True,
            "limit_up_like": True,
            "upper_shadow_pct": 0.05,
            "price_status": "ok",
        }

        result = score_signal(row, strategy_version="v2")

        self.assertEqual(result["push_level"], "观察池")
        self.assertFalse(result["is_pushed"])
        self.assertIn("封板不可买", result["suggested_action"])

    def test_v3_absorbs_tail_and_winner_rules(self) -> None:
        row = {
            "rank": 18,
            "day_return_pct": 10.02,
            "close_position": 0.96,
            "volume_ratio_5": 1.02,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 16.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 3,
            "one_word_like": False,
            "limit_up_like": True,
            "upper_shadow_pct": 0.12,
            "price_status": "ok",
        }

        v3 = score_signal(row, strategy_version="v3")
        v4 = score_signal(row, strategy_version="v4")

        self.assertGreaterEqual(v3["emotion_score"], v4["emotion_score"])
        self.assertTrue(v3["is_pushed"])
        self.assertIn("v3", "".join(v3["reasons"]))
        self.assertIn("v4", "".join(v3["reasons"]))
        self.assertEqual(strategy_default_metric_label("v2"), "次日收盘收益")
        self.assertEqual(strategy_capture_priority("v2"), ["intraday_0950", "intraday_1030", "intraday_1430", "post_close"])
        self.assertEqual(strategy_default_metric_label("v3"), "次日收盘收益")
        self.assertEqual(strategy_thresholds("v3", min_score=60), (74.0, 88.0, 102.0))
        self.assertEqual(strategy_capture_priority("v3"), ["intraday_1430", "post_close"])

    def test_v4_promotes_high_win_shape_and_uses_post_close_defaults(self) -> None:
        row = {
            "rank": 18,
            "day_return_pct": 10.01,
            "close_position": 0.96,
            "volume_ratio_5": 1.02,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 16.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 3,
            "one_word_like": False,
            "limit_up_like": True,
            "upper_shadow_pct": 0.12,
            "price_status": "ok",
        }

        v2 = score_signal(row, strategy_version="v2")
        v4 = score_signal(row, strategy_version="v4")

        self.assertGreater(v4["emotion_score"], v2["emotion_score"])
        self.assertEqual(strategy_default_metric_label("v4"), "5日收益")
        self.assertEqual(strategy_thresholds("v4", min_score=60), (74.0, 88.0, 102.0))
        self.assertEqual(strategy_capture_priority("v4"), ["post_close"])
        self.assertIn("v4", "".join(v4["reasons"]))

    def test_v4_penalizes_late_weak_shape(self) -> None:
        row = {
            "rank": 72,
            "day_return_pct": -1.2,
            "close_position": 0.58,
            "volume_ratio_5": 1.15,
            "pre5_return_pct": -2.5,
            "dist_ma20_pct": 10.0,
            "consecutive_days": 5,
            "appearance_count": 5,
            "rank_change": -14,
            "one_word_like": False,
            "limit_up_like": False,
            "upper_shadow_pct": 0.18,
            "price_status": "ok",
        }

        v2 = score_signal(row, strategy_version="v2")
        v4 = score_signal(row, strategy_version="v4")

        self.assertLess(v4["emotion_score"], v2["emotion_score"])
        self.assertIn("v4", "".join(v4["risks"]))


if __name__ == "__main__":
    unittest.main()
