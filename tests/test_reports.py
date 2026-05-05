from __future__ import annotations

import unittest

import pandas as pd

from src.reports import _fill_missing_market_context, build_latest_push, build_rule_evaluation, build_strong_recap


class ReportsMarketAuditTest(unittest.TestCase):
    def test_build_latest_push_keeps_market_audit_columns(self) -> None:
        signal_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-26",
                    "rank": 3,
                    "code": "000001",
                    "name": "Older",
                    "push_level": "普通观察",
                    "emotion_score": 60,
                    "is_pushed": True,
                },
                {
                    "signal_date": "2026-04-27",
                    "rank": 1,
                    "code": "000002",
                    "name": "Latest",
                    "push_level": "强推观察",
                    "emotion_score": 95,
                    "is_pushed": True,
                    "day_return_pct": 8.0,
                    "pre5_return_pct": 12.0,
                    "market_regime": "震荡",
                    "market_1d_pct": 0.06,
                    "market_5d_pct": -0.09,
                    "market_price_date": "2026-04-27",
                    "market_lag_days": 0,
                    "market_source": "common_index_cache",
                    "relative_1d_pct": 7.94,
                    "relative_5d_pct": 12.09,
                    "close_position": 0.95,
                    "volume_ratio_5": 1.2,
                    "dist_ma20_pct": 15.0,
                    "consecutive_days": 2,
                    "rank_change": 12,
                    "capture_type": "post_close",
                    "snapshot_time": "2026-04-27 15:00:00",
                    "reasons": "测试原因",
                    "risks": "-",
                    "suggested_action": "测试动作",
                },
            ]
        )

        result = build_latest_push(signal_df)

        self.assertEqual(result.iloc[0]["market_price_date"], "2026-04-27")
        self.assertEqual(int(result.iloc[0]["market_lag_days"]), 0)
        self.assertEqual(result.iloc[0]["market_source"], "common_index_cache")

    def test_fill_missing_market_context_backfills_market_audit_columns(self) -> None:
        base = pd.DataFrame(
            [
                {
                    "strategy_date": "2026-04-27",
                    "training_date": "2026-04-22",
                    "capture_type": "post_close",
                    "snapshot_time": "2026-04-27 15:00:00",
                    "code": "000001",
                    "market_regime": "震荡",
                    "market_1d_pct": 0.06,
                    "market_5d_pct": -0.09,
                    "market_price_date": None,
                    "market_lag_days": None,
                    "market_source": None,
                    "relative_1d_pct": 4.0,
                    "relative_5d_pct": 6.0,
                }
            ]
        )
        source = pd.DataFrame(
            [
                {
                    "strategy_date": "2026-04-27",
                    "training_date": "2026-04-22",
                    "capture_type": "post_close",
                    "snapshot_time": "2026-04-27 15:00:00",
                    "code": "000001",
                    "market_regime": "震荡",
                    "market_1d_pct": 0.06,
                    "market_5d_pct": -0.09,
                    "market_price_date": "2026-04-27",
                    "market_lag_days": 0,
                    "market_source": "common_index_cache",
                    "relative_1d_pct": 4.0,
                    "relative_5d_pct": 6.0,
                }
            ]
        )

        result = _fill_missing_market_context(base, source)

        self.assertEqual(result.iloc[0]["market_price_date"], "2026-04-27")
        self.assertEqual(result.iloc[0]["market_lag_days"], 0)
        self.assertEqual(result.iloc[0]["market_source"], "common_index_cache")

    def test_build_rule_evaluation_adds_tail_metrics(self) -> None:
        signal_df = pd.DataFrame(
            [
                {"signal_date": "2026-04-10", "code": "000001", "push_level": "强推观察", "rank": 2, "consecutive_days": 1, "is_pushed": True},
                {"signal_date": "2026-04-10", "code": "000002", "push_level": "普通观察", "rank": 33, "consecutive_days": 2, "is_pushed": True},
            ]
        )
        followup_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-10",
                    "code": "000001",
                    "push_level": "强推观察",
                    "rank": 2,
                    "consecutive_days": 1,
                    "is_pushed": True,
                    "tail_next_close_pct": 4.0,
                    "settled_tail_next_day": True,
                    "return_3d_pct": 6.0,
                    "settled_3d": True,
                },
                {
                    "signal_date": "2026-04-10",
                    "code": "000002",
                    "push_level": "普通观察",
                    "rank": 33,
                    "consecutive_days": 2,
                    "is_pushed": True,
                    "tail_next_close_pct": -2.0,
                    "settled_tail_next_day": True,
                    "return_3d_pct": None,
                    "settled_3d": False,
                },
            ]
        )

        result = build_rule_evaluation(signal_df, followup_df, strategy_version="v3")
        row = result[result["group_value"].eq("强推观察")].iloc[0]

        self.assertEqual(int(row["valid_tail_next_close"]), 1)
        self.assertEqual(float(row["avg_tail_next_close"]), 4.0)
        self.assertEqual(float(row["win_rate_tail_next_close"]), 100.0)

    def test_build_strong_recap_uses_tail_metrics_for_v3(self) -> None:
        followup_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-10",
                    "code": "000001",
                    "name": "TailWinner",
                    "rank": 5,
                    "push_level": "普通观察",
                    "emotion_score": 62,
                    "observed_days": 1,
                    "tail_next_close_pct": 16.0,
                    "tail_next_max_gain_pct": 18.0,
                    "latest_return_pct": 16.0,
                }
            ]
        )

        result = build_strong_recap(followup_df, threshold_pct=15, strategy_version="v3")

        self.assertEqual(len(result), 1)
        self.assertEqual(float(result.iloc[0]["best_return_available"]), 18.0)


if __name__ == "__main__":
    unittest.main()
