from __future__ import annotations

import unittest

import pandas as pd

from unittest.mock import patch

from src.reports import _fill_missing_market_context, build_latest_push, build_reports, build_rule_evaluation, build_strong_recap


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

    def test_build_latest_push_excludes_observation_pool_rows(self) -> None:
        signal_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-04-27",
                    "rank": 1,
                    "code": "000003",
                    "name": "Seal",
                    "push_level": "观察池",
                    "emotion_score": 99,
                    "is_pushed": False,
                }
            ]
        )

        result = build_latest_push(signal_df)

        self.assertTrue(result.empty)

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

    @patch("src.reports.write_csv")
    @patch("src.reports.run_backtest_service")
    @patch("src.reports.build_lesson_evaluation")
    @patch("src.reports.build_strong_recap")
    @patch("src.reports.build_fast_strategy_audit")
    @patch("src.reports.update_fast_strategy_history")
    @patch("src.reports.build_fast_strategy")
    def test_build_reports_light_mode_skips_heavy_reports(
        self,
        mock_build_fast_strategy,
        mock_update_fast_strategy_history,
        mock_build_fast_strategy_audit,
        mock_build_strong_recap,
        mock_build_lesson_evaluation,
        mock_run_backtest_service,
        mock_write_csv,
    ) -> None:
        signal_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-08",
                    "code": "000001",
                    "name": "Light",
                    "rank": 1,
                    "push_level": "强推观察",
                    "emotion_score": 90,
                    "is_pushed": True,
                    "capture_type": "intraday_0950",
                    "snapshot_time": "2026-05-08 09:50:00",
                }
            ]
        )
        followup_df = pd.DataFrame(
            [
                {
                    "signal_date": "2026-05-01",
                    "code": "000001",
                    "observed_days": 3,
                    "latest_return_pct": 8.0,
                    "settled_3d": True,
                }
            ]
        )
        fast_strategy_df = pd.DataFrame(
            [
                {
                    "strategy_date": "2026-05-08",
                    "training_date": "2026-05-01",
                    "capture_type": "intraday_0950",
                    "snapshot_time": "2026-05-08 09:50:00",
                    "code": "000001",
                    "name": "Light",
                    "fast_score": 88.0,
                    "rank": 1,
                }
            ]
        )
        mock_build_fast_strategy.return_value = fast_strategy_df
        mock_update_fast_strategy_history.return_value = fast_strategy_df

        result = build_reports(
            signal_df=signal_df,
            followup_df=followup_df,
            strategy_version="v2",
            light_mode=True,
        )

        mock_build_fast_strategy.assert_called_once()
        mock_update_fast_strategy_history.assert_called_once()
        mock_build_fast_strategy_audit.assert_not_called()
        mock_build_strong_recap.assert_not_called()
        mock_build_lesson_evaluation.assert_not_called()
        mock_run_backtest_service.assert_not_called()
        self.assertEqual(result["report_mode"], "light")
        self.assertEqual(result["fast_strategy_rows"], 1)
        self.assertEqual(result["latest_push_rows"], 1)
        self.assertEqual(result["fast_strategy_audit_rows"], 0)
        self.assertEqual(result["rule_evaluation_rows"], 0)
        self.assertTrue(mock_write_csv.called)


if __name__ == "__main__":
    unittest.main()
