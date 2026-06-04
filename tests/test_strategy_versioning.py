from __future__ import annotations

import unittest

import pandas as pd

from src.paths import followups_csv_for, signals_csv_for
from src.signals import CLOSE_STRENGTH_CONFIRM_LEVEL, apply_v1_daily_push_limit, build_signals, score_signal
from src.strategy_profiles import (
    available_strategy_versions,
    strategy_capture_priority,
    strategy_default_metric_label,
    strategy_thresholds,
    strategy_version_for_capture_type,
    visible_strategy_versions,
)


class StrategyVersioningTest(unittest.TestCase):
    def test_visible_versions_hide_v4_but_available_versions_keep_it(self) -> None:
        settings = {
            "strategy_versions": ["v1", "v2", "v3", "v4"],
            "visible_strategy_versions": ["v1", "v2", "v3"],
        }

        self.assertEqual(available_strategy_versions(settings), ["v1", "v2", "v3", "v4"])
        self.assertEqual(visible_strategy_versions(settings), ["v1", "v2", "v3"])

    def test_versioned_paths_keep_v1_and_split_other_versions(self) -> None:
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
        self.assertEqual(v1["push_level"], CLOSE_STRENGTH_CONFIRM_LEVEL)
        self.assertTrue(v1["is_pushed"])
        self.assertFalse(v2["is_pushed"])
        self.assertIn("v2", str(v1["reasons"]))
        self.assertIn("v2", str(v2["reasons"]))
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
        self.assertIn("首次上榜强势提权", str(first["reasons"]))

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
        self.assertIn("一字板不可买", str(result["suggested_action"]))

    def test_limit_up_like_moves_tail_version_to_observation_pool(self) -> None:
        row = {
            "rank": 6,
            "day_return_pct": 10.02,
            "close_position": 0.99,
            "volume_ratio_5": 1.02,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 16.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 2,
            "one_word_like": False,
            "limit_up_like": True,
            "upper_shadow_pct": 0.05,
            "price_status": "ok",
        }

        result = score_signal(row, strategy_version="v3")

        self.assertEqual(result["push_level"], "观察池")
        self.assertFalse(result["is_pushed"])
        self.assertIn("涨停封板不可买", str(result["suggested_action"]))

    def test_v1_daily_push_limit_keeps_only_best_five_candidates(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "signal_date": "2026-05-11",
                    "code": f"00000{index}",
                    "rank": index + 1,
                    "price_status": "ok",
                    "emotion_score": 100 - index,
                    "is_pushed": True,
                    "push_level": "强推观察",
                    "day_return_pct": 8.0,
                    "close_position": 0.9,
                    "volume_ratio_5": 1.0,
                    "pre5_return_pct": 12.0,
                    "dist_ma20_pct": 10.0,
                    "consecutive_days": 2,
                    "rank_change": 1,
                    "market_score": 65.0,
                    "market_5d_pct": 2.0,
                    "reasons": "-",
                    "risks": "-",
                    "suggested_action": "-",
                    "strategy_version": "v1",
                }
            )

        result = apply_v1_daily_push_limit(pd.DataFrame(rows), max_pushed=5)

        self.assertEqual(int(result["is_pushed"].astype(bool).sum()), 5)
        demoted = result[~result["is_pushed"].astype(bool)].iloc[0]
        self.assertFalse(bool(demoted["is_pushed"]))
        self.assertEqual(demoted["push_level"], "观察池")
        self.assertIn("v1每日精选未入前5", str(demoted["risks"]))

    def test_v3_uses_tail_rules_without_v4_overlay(self) -> None:
        row = {
            "rank": 18,
            "day_return_pct": 7.59,
            "close_position": 0.85,
            "volume_ratio_5": 1.02,
            "pre5_return_pct": 18.0,
            "dist_ma20_pct": 16.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 3,
            "one_word_like": False,
            "limit_up_like": False,
            "upper_shadow_pct": 0.12,
            "price_status": "ok",
        }

        v3 = score_signal(row, strategy_version="v3")

        self.assertTrue(v3["is_pushed"])
        self.assertIn("v3", str(v3["reasons"]))
        self.assertNotIn("v4", str(v3["reasons"]))
        self.assertEqual(strategy_default_metric_label("v2"), "次日收盘收益")
        self.assertEqual(strategy_capture_priority("v2"), ["intraday_0950", "intraday_1030", "intraday_1430", "post_close"])
        self.assertEqual(strategy_default_metric_label("v3"), "次日收盘收益")
        self.assertEqual(strategy_thresholds("v3", min_score=60), (74.0, 88.0, 102.0))
        self.assertEqual(strategy_capture_priority("v3"), ["intraday_1430", "post_close"])

    def test_v2_rewards_executable_recap_shape(self) -> None:
        strong_row = {
            "rank": 28,
            "day_return_pct": 6.3,
            "close_position": 0.83,
            "volume_ratio_5": 1.05,
            "pre5_return_pct": 15.0,
            "dist_ma20_pct": 13.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 2,
            "one_word_like": False,
            "limit_up_like": False,
            "upper_shadow_pct": 0.1,
            "price_status": "ok",
        }
        weak_row = dict(
            strong_row,
            rank=63,
            day_return_pct=-1.5,
            close_position=0.52,
            volume_ratio_5=1.25,
            pre5_return_pct=23.0,
            consecutive_days=5,
            rank_change=-12,
        )

        strong = score_signal(strong_row, strategy_version="v2")
        weak = score_signal(weak_row, strategy_version="v2")

        self.assertGreater(strong["emotion_score"], weak["emotion_score"])
        self.assertIn("v2", str(strong["reasons"]))
        self.assertIn("v2", str(weak["risks"]))

    def test_v2_penalizes_nonlimit_high_chase_shape(self) -> None:
        clean_row = {
            "rank": 12,
            "day_return_pct": 6.8,
            "close_position": 0.88,
            "volume_ratio_5": 1.18,
            "pre5_return_pct": 14.0,
            "dist_ma20_pct": 12.0,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 18,
            "one_word_like": False,
            "limit_up_like": False,
            "upper_shadow_pct": 0.08,
            "price_status": "ok",
        }
        stretched_row = dict(
            clean_row,
            rank=36,
            day_return_pct=8.8,
            close_position=0.98,
            volume_ratio_5=1.82,
            pre5_return_pct=28.0,
        )

        clean = score_signal(clean_row, strategy_version="v2")
        stretched = score_signal(stretched_row, strategy_version="v2")

        self.assertGreater(clean["emotion_score"], stretched["emotion_score"])
        self.assertIn("v2", str(stretched["risks"]))

    def test_capture_type_routes_to_expected_strategy_version(self) -> None:
        self.assertEqual(strategy_version_for_capture_type("post_close"), "v1")
        self.assertEqual(strategy_version_for_capture_type("intraday_0935"), "v2")
        self.assertEqual(strategy_version_for_capture_type("intraday_0950"), "v2")
        self.assertEqual(strategy_version_for_capture_type("intraday_1030"), "v2")
        self.assertEqual(strategy_version_for_capture_type("intraday_1430"), "v3")
        self.assertIsNone(strategy_version_for_capture_type("tests"))

    def test_v4_rewards_technical_pullback_and_uses_post_close_defaults(self) -> None:
        row = {
            "rank": 9,
            "day_return_pct": 2.8,
            "close_position": 0.84,
            "volume_ratio_5": 0.88,
            "pre3_return_pct": 2.5,
            "pre5_return_pct": 8.0,
            "dist_ma5_pct": 1.2,
            "dist_ma10_pct": 2.9,
            "dist_ma20_pct": 8.5,
            "upper_shadow_pct": 0.08,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 5,
            "one_word_like": False,
            "limit_up_like": False,
            "price_status": "ok",
        }

        v1 = score_signal(row, strategy_version="v1")
        v4 = score_signal(row, strategy_version="v4")

        self.assertGreater(v4["emotion_score"], v1["emotion_score"])
        self.assertEqual(strategy_default_metric_label("v4"), "次日收盘收益")
        self.assertEqual(strategy_thresholds("v4", min_score=60), (80.0, 94.0, 108.0))
        self.assertEqual(strategy_capture_priority("v4"), ["post_close"])
        self.assertIn("v4", str(v4["reasons"]))

    def test_v4_penalizes_stretched_and_overheated_shape(self) -> None:
        clean_row = {
            "rank": 9,
            "day_return_pct": 2.8,
            "close_position": 0.84,
            "volume_ratio_5": 0.88,
            "pre3_return_pct": 2.5,
            "pre5_return_pct": 8.0,
            "dist_ma5_pct": 1.2,
            "dist_ma10_pct": 2.9,
            "dist_ma20_pct": 8.5,
            "upper_shadow_pct": 0.08,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 5,
            "one_word_like": False,
            "limit_up_like": False,
            "price_status": "ok",
        }
        stretched_row = {
            "rank": 56,
            "day_return_pct": 8.2,
            "close_position": 0.68,
            "volume_ratio_5": 2.4,
            "pre3_return_pct": 18.0,
            "pre5_return_pct": 34.0,
            "dist_ma5_pct": 6.8,
            "dist_ma10_pct": 12.0,
            "dist_ma20_pct": 30.0,
            "upper_shadow_pct": 0.42,
            "consecutive_days": 4,
            "appearance_count": 4,
            "rank_change": -6,
            "one_word_like": False,
            "limit_up_like": False,
            "price_status": "ok",
        }

        clean = score_signal(clean_row, strategy_version="v4")
        stretched = score_signal(stretched_row, strategy_version="v4")
        base = score_signal(stretched_row, strategy_version="v1")

        self.assertGreater(clean["emotion_score"], stretched["emotion_score"])
        self.assertLessEqual(stretched["emotion_score"], base["emotion_score"])
        self.assertIn("v4", str(stretched["risks"]))

    def test_v4_event_penalty_deducts_without_forcing_observation_pool(self) -> None:
        base_row = {
            "rank": 11,
            "day_return_pct": 3.4,
            "close_position": 0.82,
            "volume_ratio_5": 0.92,
            "pre3_return_pct": 3.0,
            "pre5_return_pct": 9.0,
            "dist_ma5_pct": 1.0,
            "dist_ma10_pct": 2.5,
            "dist_ma20_pct": 8.0,
            "upper_shadow_pct": 0.1,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 6,
            "one_word_like": False,
            "limit_up_like": False,
            "price_status": "ok",
        }
        event_row = dict(base_row, announcement_summary="公司收到监管问询函，股东披露减持计划")

        base = score_signal(base_row, strategy_version="v4")
        event = score_signal(event_row, strategy_version="v4")

        self.assertGreater(base["emotion_score"], event["emotion_score"])
        self.assertIn("v4事件风险扣分", str(event["risks"]))
        self.assertNotEqual(event["push_level"], "观察池")

    def test_v4_flow_chip_and_dragon_tiger_can_raise_score(self) -> None:
        base_row = {
            "rank": 18,
            "day_return_pct": 2.6,
            "close_position": 0.81,
            "volume_ratio_5": 0.94,
            "pre3_return_pct": 2.7,
            "pre5_return_pct": 8.8,
            "dist_ma5_pct": 1.4,
            "dist_ma10_pct": 2.9,
            "dist_ma20_pct": 8.1,
            "upper_shadow_pct": 0.09,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 4,
            "one_word_like": False,
            "limit_up_like": False,
            "price_status": "ok",
        }
        enhanced_row = dict(
            base_row,
            capital_flow_signal=4.6,
            concentration_90=13.2,
            profit_ratio=63.0,
            dragon_tiger_positive=True,
        )

        base = score_signal(base_row, strategy_version="v4")
        enhanced = score_signal(enhanced_row, strategy_version="v4")

        self.assertGreater(enhanced["emotion_score"], base["emotion_score"])
        self.assertIn("v4主力资金强净流入", str(enhanced["reasons"]))
        self.assertIn("v4筹码资金共振", str(enhanced["reasons"]))

    def test_v4_positive_and_negative_event_contexts_are_distinct(self) -> None:
        base_row = {
            "rank": 34,
            "day_return_pct": 1.8,
            "close_position": 0.76,
            "volume_ratio_5": 0.82,
            "pre3_return_pct": 1.5,
            "pre5_return_pct": 5.8,
            "dist_ma5_pct": 0.6,
            "dist_ma10_pct": 1.6,
            "dist_ma20_pct": 6.2,
            "upper_shadow_pct": 0.08,
            "consecutive_days": 2,
            "appearance_count": 1,
            "rank_change": 4,
            "one_word_like": False,
            "limit_up_like": False,
            "price_status": "ok",
        }
        positive_row = dict(base_row, announcement_summary="公司签订重大合同")
        caution_row = dict(base_row, announcement_summary="公司发布异常波动和风险提示公告")
        negative_row = dict(base_row, announcement_summary="公司收到监管问询函，股东披露减持计划")

        positive = score_signal(positive_row, strategy_version="v4")
        caution = score_signal(caution_row, strategy_version="v4")
        negative = score_signal(negative_row, strategy_version="v4")

        self.assertGreater(positive["emotion_score"], caution["emotion_score"])
        self.assertGreater(caution["emotion_score"], negative["emotion_score"])
        self.assertIn("v4事件催化加分", str(positive["reasons"]))
        self.assertIn("v4事件谨慎扣分", str(caution["risks"]))

    def test_build_signals_preserves_v4_strategy_version(self) -> None:
        feature_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-14",
                    "code": "600001",
                    "rank": 10,
                    "day_return_pct": 3.5,
                    "close_position": 0.82,
                    "volume_ratio_5": 0.95,
                    "pre5_return_pct": 8.0,
                    "dist_ma5_pct": 1.2,
                    "dist_ma10_pct": 2.7,
                    "dist_ma20_pct": 8.4,
                    "upper_shadow_pct": 0.1,
                    "consecutive_days": 2,
                    "appearance_count": 1,
                    "rank_change": 5,
                    "one_word_like": False,
                    "limit_up_like": False,
                    "price_status": "ok",
                }
            ]
        )

        result = build_signals(feature_df=feature_df, strategy_version="v4")

        self.assertEqual(result.iloc[0]["strategy_version"], "v4")


if __name__ == "__main__":
    unittest.main()
